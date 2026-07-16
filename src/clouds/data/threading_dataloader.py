import itertools
import queue
import threading

import torch
import torch.utils.data._utils as _utils
from torch._utils import ExceptionWrapper
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch_geometric.loader.dataloader import DataLoader


class _ThreadCompatQueue(queue.Queue):
    """Multiprocessing compatibility shim

    queue.Queue that also satisfies the small slice of the
    multiprocessing.Queue interface that torch's _worker_loop calls
    during shutdown (cancel_join_thread, close). Both are no-ops here:
    queue.Queue has no feeder thread or OS pipe to flush/close."""

    def cancel_join_thread(self):
        pass

    def close(self):
        pass


class _MultiThreadingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler."""

    def __init__(self, loader):
        super().__init__(loader)

        self._prefetch_factor = loader.prefetch_factor
        self._num_workers = loader.num_workers

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        self._worker_init_fn = loader.worker_init_fn

        # Queue for inter-thread communication
        self._worker_result_queue = _ThreadCompatQueue()
        self._workers_done_event = threading.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = _ThreadCompatQueue()
            self._index_queues.append(index_queue)

            # Initialize the worker threads
            worker_thread = threading.Thread(
                target=_utils.worker._worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                    self._shared_seed,
                ),
            )
            worker_thread.daemon = True
            worker_thread.start()
            self._workers.append(worker_thread)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._data_queue = _ThreadCompatQueue()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    torch.cuda.current_device(),
                    self._pin_memory_thread_done_event,
                    self._pin_memory_device,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        """Initialize/reset state for the iterator."""
        super()._reset(loader, first_iter)

        self._send_idx = 0  # The index of the next task to be sent to workers
        self._rcvd_idx = 0  # The index of the next task to be received
        self._task_info = {}  # Mapping from task index to (worker_id, data) tuples
        self._tasks_outstanding = 0  # Count of tasks sent but not yet received
        self._shutdown = False

        # Set workers' status: True if the worker is active, False if it has exhausted its dataset
        self._workers_status = [True for _ in range(self._num_workers)]

        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))

        # Prefetch the first batches of data
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _shutdown_workers(self):
        """Shutdown worker threads instead of processes."""
        if not self._shutdown:
            self._shutdown = True
            try:
                # Signal the workers to stop
                self._workers_done_event.set()
                for index_queue in self._index_queues:
                    index_queue.put(None)  # Signal termination to each worker

                # Join the worker threads
                for worker in self._workers:
                    worker.join()

                if hasattr(self, '_pin_memory_thread'):
                    self._pin_memory_thread_done_event.set()
                    self._pin_memory_thread.join()
            finally:
                pass

    def _try_put_index(self):
        """Tries to put the next index into a worker's queue."""
        # Ensure the number of outstanding tasks is less than the max prefetch amount
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            # Get the next index to process from the sampler
            index = self._next_index()
        except StopIteration:
            # No more indices to fetch
            return

        # Find the next active worker in a round-robin fashion
        for _ in range(self._num_workers):
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # No active worker found, return without assigning new work
            return

        # Place the index in the selected worker's queue for processing
        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)  # Record the worker handling this task
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except queue.Empty:
            return (False, None)

    def _next_data(self):
        while True:
            # Check if the next sample has already been generated
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]

                # If the data for the current index has already been fetched
                if len(info) == 2:
                    data = self._task_info.pop(self._rcvd_idx)[1]
                    return self._process_data(data)

                # If worker is still active or data is not yet fetched
                if self._workers_status[worker_id]:
                    break

                # If not, remove the entry for this index
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # If we have iterated through all indices and there's nothing left
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Fetch the next piece of data from the queue
            idx, data = self._get_data_from_queue()
            self._tasks_outstanding -= 1

            # If the fetched data is out of order, store it temporarily
            if idx != self._rcvd_idx:
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _get_data_from_queue(self):
        """Helper function to fetch data from the data queue."""
        while True:
            # Try to get the data from the queue
            success, data = self._try_get_data()
            if success:
                return data

    def _process_data(self, data):
        """Processes data after it has been fetched by a worker."""
        # Increment the index of received tasks
        self._rcvd_idx += 1

        # Try to add new indices to the queue for the next batch
        self._try_put_index()

        # If the data is an ExceptionWrapper, raise the exception
        if isinstance(data, ExceptionWrapper):
            data.reraise()  # This will raise the original exception from the worker

        return data  # Return the data fetched from the worker


class ThreadingDataLoader(DataLoader):
    """Drop-in DataLoader that uses thread workers instead of process workers."""

    def _get_iterator(self):
        if self.num_workers == 0:
            return super()._get_iterator()  # fall back to single-process iter
        return _MultiThreadingDataLoaderIter(self)

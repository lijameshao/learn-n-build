import asyncio
from asyncio import AbstractEventLoop
import time
from time import sleep

CHUNK_SIZE = 2


def io_bound_task(task_input: int):
    print(f"io bound task input {task_input}")
    print(f"Time: {time.time() - start}")
    sleep(1)
    return task_input


def get_async_task(loop: AbstractEventLoop, task: int):
    return loop.run_in_executor(None, io_bound_task, task)


def async_runner(loop: AbstractEventLoop, tasks_input: list[int]):
    chunked_tasks = chunks(tasks_input, CHUNK_SIZE)
    return [
        loop.run_in_executor(None, io_bound_task, task)
        for chunked_task in chunked_tasks
        for task in chunked_task
    ]


async def main():
    tasks = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11]]
    loop = asyncio.get_event_loop()

    async_tasks = [async_runner(loop, task) for task in tasks]
    results = [await asyncio.gather(*async_task) for async_task in async_tasks]

    print(results)
    print("Done!")
    return results


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    start = time.time()
    results = asyncio.run(main())
    print(results)

"""
Output:

io bound task input 1
Time: 0.001466989517211914
io bound task input 2
Time: 0.0016179084777832031
io bound task input 3
Time: 0.0017130374908447266
io bound task input 4
Time: 0.0018892288208007812
io bound task input 5
Time: 0.001987934112548828
io bound task input 6
Time: 0.002223968505859375
io bound task input 7
Time: 0.0023050308227539062
io bound task input 8
Time: 0.0029990673065185547
io bound task input 9
io bound task input 11
Time: 0.00328826904296875
Time: 0.0033080577850341797
io bound task input 10
Time: 0.0034341812133789062
[[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11]]
Done!
[[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11]]
"""

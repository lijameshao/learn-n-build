import asyncio
import time
from time import sleep

CHUNK_SIZE = 2


def io_bound_task(task_input: int):
    print(f"io bound task input {task_input}")
    print(f"Time: {time.time() - start}")
    sleep(1)
    return task_input


async def async_runner(task_input: list[int]):
    loop = asyncio.get_event_loop()
    chunked_task = chunks(task_input, CHUNK_SIZE)
    tasks = []
    for c in chunked_task:
        tasks.append(loop.run_in_executor(None, io_bound_task, c))
    chunked_results = await asyncio.gather(*tasks)
    results = [r for chunked_result in chunked_results for r in chunked_result]
    return results


async def main():
    tasks = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11]]

    task_results = []
    for task in tasks:
        task_result = await async_runner(task)
        task_results.append(task_result)
    print(task_results)
    print("Done!")


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    start = time.time()
    result = asyncio.run(main())

"""
Output:

io bound task input [1, 2]
Time: 0.0014040470123291016
io bound task input [3, 4]
Time: 0.001605987548828125
io bound task input [5, 6]
Time: 0.0017092227935791016
io bound task input [7, 8]
Time: 1.006925106048584
io bound task input [9]
Time: 1.0069482326507568
io bound task input [10, 11]
Time: 2.013857126235962
[[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11]]
Done!

"""

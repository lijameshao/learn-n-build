import asyncio
import time
from time import sleep


def sync_task() -> list[int]:
    print("Start sync task")
    sleep(2)
    print("Sync task done")
    print(f"Time: {time.time() - start}")
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


async def async_task(sync_task_output: list[int]):

    subtasks = [io_bound_task(num) for num in sync_task_output]

    subtask_result_list = await asyncio.gather(*subtasks)
    subtask_result = sum(subtask_result_list)

    return subtask_result


async def io_bound_task(task_input: int) -> int:
    print(f"io bound task input {task_input}")
    print(f"Time: {time.time() - start}")
    await asyncio.sleep(1)
    return task_input


async def main(sync_outputs: list[list[int]]):

    temp_results = []
    for sync_output in sync_outputs:
        temp_result = await async_task(sync_output)
        temp_results.append(temp_result)

    print(f"temp_results: {temp_results}")
    result = await async_task(temp_results)

    return result


if __name__ == "__main__":
    start = time.time()
    sync_outputs = sync_task()
    result = asyncio.run(main(sync_outputs))
    print(f"Result: {result}")
    print("Done")

# -*- coding: utf-8 -*-

import asyncio
from asyncio import AbstractEventLoop, Future
import time
import random

def io_bound_task(t: int):
    time.sleep(1)
    r = random.randint(0, 1)
    time_elapsed = time.time() - start
    if r == 1:
        print(f"Task: {t}")
        print(f"Time: {time_elapsed}")
        raise Exception(f"Random exception, task {t}, time: {time_elapsed}")
    print(f"Task: {t}")
    print(f"Time: {time_elapsed}")
    return t+1


async def async_task(
    loop: AbstractEventLoop, input_tasks: list[int]
) -> list[int]:
    
    lst_of_tasks = []
    for t in input_tasks:
        nested_task = [t,t]
        lst_of_tasks.append(
            [loop.run_in_executor(None, io_bound_task, t) for t in nested_task]
        )

    future_results: list[list[Future]] = [
        await asyncio.gather(*tasks, return_exceptions=True) for tasks in lst_of_tasks
    ]
    results = []
    for lst_of_nested_results in future_results:
        count = 0
        for r in lst_of_nested_results:
            if isinstance(r, Exception):
                print(r)
                continue
            count += r
        results.append(count)
    return results


async def main():
    input_task = [1,2,3,4]
    loop = asyncio.get_event_loop()
    results = await async_task(loop, input_task)
    output: list[int] = []
    for i in range(len(results)):
        output.append(
            {
                "input": input_task[i],
                "output": results[i],
            }
        )
    return output


if __name__ == "__main__":
    start = time.time()
    results = asyncio.run(main())
    time_elapsed = time.time() - start
    print(f"Final time: {time_elapsed}")
    print(f"Final results: {results}")

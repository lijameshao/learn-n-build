import asyncio
from asyncio import AbstractEventLoop
import time
from time import sleep

CHUNK_SIZE = 2


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def sync_task(s: str) -> str:
    print(f"Start sync task for {s}")
    sleep(2)
    print(f"Sync task done for {s}")
    print(f"Time: {time.time() - start}")
    return s


async def task_with_chunking(loop: AbstractEventLoop, texts: list[str]):
    chunked_tasks = []
    for text in texts:
        tokenised_text = text.split(" ")
        chunked_tokens = chunks(tokenised_text, CHUNK_SIZE)
        chunked_texts = [" ".join(tokens_list) for tokens_list in chunked_tokens]
        chunked_tasks.append(
            [loop.run_in_executor(None, sync_task, t) for t in chunked_texts]
        )

    results = [await asyncio.gather(*tasks) for tasks in chunked_tasks]
    results = [" ".join(lst) for lst in results]
    return results


async def chapter_extraction(loop: AbstractEventLoop, texts: list[str]) -> list[str]:
    results = await task_with_chunking(loop, texts)
    return results


async def main():
    loop = asyncio.get_event_loop()
    concatenated_results = []
    lst_of_lst_of_tasks = [["1", "1 1", "1 1 1 1"], ["2", "22 22"], ["3", "33 33"]]

    coros = [chapter_extraction(loop, c) for c in lst_of_lst_of_tasks]
    lst_of_lst_of_results = await asyncio.gather(*coros)

    for lst_of_results in lst_of_lst_of_results:
        concatenated_result = " ".join(lst_of_results)
        concatenated_results.append(concatenated_result)

    final_result = await task_with_chunking(loop, concatenated_results)
    return final_result


if __name__ == "__main__":
    start = time.time()
    results = asyncio.run(main())
    print(results)

"""
Output:

Start sync task for 1
Start sync task for 1 1
Start sync task for 1 1
Start sync task for 1 1
Start sync task for 2
Start sync task for 22 22
Start sync task for 3
Start sync task for 33 33
Sync task done for 1
Time: 2.0066659450531006
Sync task done for 1 1
Time: 2.008683919906616
Sync task done for 33 33
Time: 2.008798122406006
Sync task done for 1 1
Time: 2.00887393951416
Sync task done for 2
Time: 2.0089640617370605
Sync task done for 3
Time: 2.0090441703796387
Sync task done for 1 1
Time: 2.00911808013916
Sync task done for 22 22
Time: 2.009290933609009
Start sync task for 1 1
Start sync task for 1 1
Start sync task for 1 1
Start sync task for 1
Start sync task for 2 22
Start sync task for 22
Start sync task for 3 33
Start sync task for 33
Sync task done for 1 1
Time: 4.017538070678711
Sync task done for 1 1
Time: 4.0182859897613525
Sync task done for 1
Time: 4.019296884536743
Sync task done for 2 22
Time: 4.01940393447876
Sync task done for 3 33
Time: 4.019485235214233
Sync task done for 1 1
Time: 4.019674062728882
Sync task done for 33
Time: 4.019942045211792
Sync task done for 22
Time: 4.020020961761475
['1 1 1 1 1 1 1', '2 22 22', '3 33 33']
"""

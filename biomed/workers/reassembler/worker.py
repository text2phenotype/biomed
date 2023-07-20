import concurrent.futures
from typing import (
    List,
    TYPE_CHECKING,
)

from biomed.reassembler.reassemble_functions import get_reassemble_function
from text2phenotype.common.log import operations_logger
from text2phenotype.tasks.mixins import RedisMethodsMixin
from text2phenotype.tasks.rmq_worker import RMQConsumerTaskWorker
from text2phenotype.tasks.task_enums import (
    TaskEnum,
    TaskStatus,
    WorkType,
)
from text2phenotype.tasks.task_info import (
    ChunkTaskInfo,
    create_task_info,
    TASK_MAPPING,
)

from biomed.biomed_env import BiomedEnv

if TYPE_CHECKING:
    from text2phenotype.tasks.work_tasks import (
        ChunkTask,
        DocumentTask,
    )


class Reassembler(RMQConsumerTaskWorker):
    QUEUE_NAME = BiomedEnv.REASSEMBLE_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.reassemble
    RESULTS_FILE_EXTENSION = None
    WORK_TYPE = WorkType.document
    NAME = 'Reassembler'
    ROOT_PATH = BiomedEnv.root_dir

    def do_work(self):
        task_result = self.init_task_result()

        with concurrent.futures.ProcessPoolExecutor(1) as executor:
            executor.submit(self.reassemble_results, self.work_task, self.storage_client).result()

        return task_result

    @staticmethod
    def reassemble_results(work_task: 'DocumentTask', storage_client):
        from biomed.reassembler import ReassemblerResultManager
        from biomed.summary.text_to_summary import add_page_numbers_to_predictions

        deid_in_operations = TaskEnum.phi_tokens in work_task.chunk_tasks
        
        if deid_in_operations:
            operations_logger.info('"DEID is in the operation set, will reassemble annotations')
        else:
            operations_logger.info(f"phi_tokens not in {work_task.chunk_tasks}, or {list(work_task.task_statuses.keys())}")

        redis = RedisMethodsMixin()
        chunks = [redis.get_task(work_type=WorkType.chunk, redis_key=chunk)
                  for chunk in work_task.chunks]

        for task_enum in work_task.chunk_tasks:
            task_info = create_task_info(task_enum)
            task_info_class: ChunkTaskInfo = task_info.__class__()

            failing_chunks = Reassembler._get_failing_chunks(chunks, task_enum)
            # make sure all other dependencies completed
            for other_dependency in task_info_class.model_dependencies:
                failing_chunks.extend(Reassembler._get_failing_chunks(chunks, other_dependency))

            if not failing_chunks:
                results_file_key = None
                if get_reassemble_function(task_enum, include_annotations=deid_in_operations) is not None:
                    chunk_mapping = task_info_class.iter_chunk_results(
                        chunks=chunks,
                        storage_client=storage_client)
                    # get mapping of enum: chunk results
                    enum_to_chunk_mapping = dict()
                    for other_req_model_chunk_task_enum in task_info_class.model_dependencies:
                        other_req_model_task_info = TASK_MAPPING[other_req_model_chunk_task_enum]
                        enum_to_chunk_mapping[
                            other_req_model_chunk_task_enum] = other_req_model_task_info.iter_chunk_results(
                            chunks,
                            storage_client)

                    with ReassemblerResultManager() as final_result:
                        final_result.reassemble_chunks(task_enum, chunk_mapping, enum_to_chunk_mapping,
                                                       deid_in_task_enums=deid_in_operations)

                        original_text = storage_client.get_content(work_task.text_file_key).decode('utf8')

                        if original_text and final_result.data:
                            add_page_numbers_to_predictions(original_text, final_result.data)

                        with final_result.to_json_stream() as stream:
                            results_file_key = task_info_class.write_document_results(
                                data=None,
                                file=stream,
                                document_id=work_task.document_id,
                                storage_client=storage_client
                            )

                    del final_result

                with redis.task_update_manager(work_task) as work_task:
                    task_status = work_task.task_statuses.setdefault(task_enum, task_info)
                    task_status.results_file_key = results_file_key
                    task_status.status = TaskStatus.completed_success

            else:
                with redis.task_update_manager(work_task) as work_task:
                    task_status = work_task.task_statuses.setdefault(task_enum, task_info)
                    task_status.status = TaskStatus.completed_failure
                    task_status.error_messages.append(f'The following chunks failed: '
                                                      f'{[c.redis_key for c in failing_chunks]}')

    @staticmethod
    def _get_failing_chunks(chunks: List['ChunkTask'], task: TaskEnum) -> List['ChunkTask']:
        failing_chunks = []
        for chunk in chunks:
            if chunk.task_statuses[task].status is not TaskStatus.completed_success:
                failing_chunks.append(chunk)

        return failing_chunks

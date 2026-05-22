import os
import sys
from clearml import Task
 
 
def main():
    required = ["CLEARML_API_HOST", "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
 
    print(f"Connecting to ClearML at {os.environ['CLEARML_API_HOST']} ...")
 
    task = Task.init(
        project_name="anonifyme-scrfd-retrain",
        task_name="scheduled-retraining-trigger",
        task_type=Task.TaskTypes.controller,
        reuse_last_task_id=False,
    )
 
    task.execute_remotely(queue_name="default", exit_process=True)
    print("✅ Retraining pipeline enqueued on ClearML queue: default")
    print(f"   Task ID : {task.id}")
    print(f"   Monitor : {os.environ['CLEARML_API_HOST']}/projects/")
    print("   The pipeline will execute when a ClearML agent comes online.")
 
 
if __name__ == "__main__":
    main()
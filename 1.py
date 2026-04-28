import executorch.backends.qualcomm.utils.utils as q_utils
import inspect

all_members = dir(q_utils)

print(f"Functions in {q_utils.__name__}:")
print("-" * 40)

for name in all_members:
    item = getattr(q_utils, name)

    if inspect.isfunction(item):
        if item.__module__ == q_utils.__name__:
            print(name)
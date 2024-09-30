import json
import os
import re

o = os.environ["KV"]
o = json.loads(o)

data = """
 {
    "vit": {
        "single-gpu": [
            {
                "test": "tests/models/vit/test_modeling_vit.py::ViTModelTest::test_foo",
                "commit": "4b7de7cf31573a3b924c3473e1b8cc913cb0f63a",
                "pr_number": null,
                "author": "ydshieh",
                "merged_by": null
            }
        ],
        "multi-gpu": []
    }
}
"""


# print(json.dumps(json.loads(data)))
# print(json.dumps(json.loads(data)).replace('"', '\\"'))

# exit(0)


# o = {
#     "YDSHIEH_SLACK_ID": "***",
# }

# with open("new_model_failures_with_bad_commit.json") as fp:
#     data = json.load(fp)

data = json.loads(data)

for model, model_result in data.items():
    for device, failed_tests in model_result.items():
        for failed_test in failed_tests:
            author = failed_test["author"].replace("-", "_").upper() + "_SLACK_ID"
            if author in o:
                author = o[author]
            failed_test["author"] = f"<@{author}>"

print(json.dumps(data).replace('"', '\\"'))




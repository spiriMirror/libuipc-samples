# Wrecking Ball

This is a multi-body simulation using libuipc.

![image](image.png)

The script shows the usage of `recovery` and `dump` functionallity of the simulation engine.

Users are able save the state of the simulation to the disk and recover it later.

Note:

- The behavior of recovery and dumping is backend dependent. 
- You can delete the `dump/` folder in the workspace to clear the cache.

## Q&A

Where is the `workspace`?

```python
workspace = './' # it's up to you where to put the workspace
engine = Engine('cuda', workspace)
```

For this sample, the workspace is [libuipc-samples/output/python/3_wrecking_balls/main.py/](../../output/python/3_wrecking_balls/main.py/), which is calculated by the `AssetDir.output_path(__file__)`.
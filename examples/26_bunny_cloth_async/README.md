# Bunny Cloth Async

This is an example of using `uipc.Future` to run physics simulation asynchronously in a separate thread to avoid blocking the main thread. 
This allows the UI to remain responsive while the physics simulation runs in the background. 

# Usage

```python
def async_run():
    world.advance()
    world.retrieve()

f = Future.launch(async_run)
# get the control immediately
if f.is_ready():
    # do something with the result
    pass
```
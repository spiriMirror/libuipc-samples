# Advanced Scene Configuration Example

This example demonstrates how to modify a scene's configuration using the `Scene.config()` method. This method returns a mutable `ConfigAttributes` that allows you to adjust various simulation parameters.

The `ConfigAttributes` has the same interface as any other geometry attribute collection like `.meta()`/`.vertices()`. You need to call `.find('ATTR_NAME')` to get a specific attribute, and use `view()` function to read or write the attribute values.

```python
config = scene.config()
dt_attr = config.find('dt')
view(dt_attr)[:] = 0.02
```

The nested structure of the json configuration is expanded as paths using '/' as a delimiter. 

```python
# Original way to set friction:
# config = Scene.default_config()
# config['contact']['friction']['enable'] = 1
friction_enable = config.find('contact/friction/enable')
view(friction_enable)[:] = 1
```

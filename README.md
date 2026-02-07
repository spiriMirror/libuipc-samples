# libuipc-samples

A sample library for Libuipc.

This library is a collection of sample programs that demonstrate how to use the Libuipc library.

## uv (recommended)

If you are using uv, you can use the following command to install the required dependencies:

```bash
uv sync
```

Note: we use pyuipc on pypi, it only supports cuda 12.8 for now, so you need to ensure you are using cuda 12.8. If you want to use other cuda versions, you need to [build pyuipc from source](https://spirimirror.github.io/libuipc-doc/build_install/).

## build from source

Before you using this library, please refer to the [Libuipc](https://github.com/spiriMirror/libuipc) to build and install the library.

To run the python interactable sample, you may need to install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

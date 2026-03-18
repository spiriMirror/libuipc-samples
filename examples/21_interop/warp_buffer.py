import uipc
import warp as wp
import uipc.adapter.warp

wp.init()

# create uipc buffer (managed by warp)
wb = uipc.adapter.warp.buffer(dtype=wp.float32, device='cuda')
print(wb.buffer_view())
wb.resize(10)
print(wb.buffer_view())

# get warp array from uipc buffer
print(wb.warp())
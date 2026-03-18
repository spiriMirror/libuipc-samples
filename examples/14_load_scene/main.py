import json
import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc 
from uipc import view
from uipc import Vector3, Vector2, Transform, Logger, Quaternion, AngleAxis, Timer
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene, SceneIO, Object
from uipc.gui import SceneGUI

from asset_dir import AssetDir

workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)
engine = Engine("cuda", workspace=workspace)
world = World(engine)

scene = SceneIO.load(f'{folder}/scene.json')

world.init(scene)
sgui = SceneGUI(scene, 'split')

Logger.set_level(Logger.Level.Warn)

ps.init()
sgui.register()
sgui.set_edge_width(1)

run = False
def on_update():
    global run
    if(imgui.Button('run & stop')):
        run = not run
        
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()


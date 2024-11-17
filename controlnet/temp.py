try:
    from ldm.modules.attention import SpatialTransformer as SpatialTransformer1
    print("ldm.modules.attention imported from:", SpatialTransformer1.__module__)
except ImportError as e:
    print("Failed to import from ldm.modules.attention:", e)

try:
    from controlnet.ldm.modules.attention import SpatialTransformer as SpatialTransformer2
    print("controlnet.ldm.modules.attention imported from:", SpatialTransformer2.__module__)
except ImportError as e:
    print("Failed to import from controlnet.ldm.modules.attention:", e)

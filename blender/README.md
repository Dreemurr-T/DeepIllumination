# Blender plugin

This only works for a special customized blender version 2.79
from https://github.com/bitsawer/blender-custom-nodes,
which allows us to make use of Python script on compositing nodes.

You could download the Windows version of modified blender in
https://github.com/bitsawer/blender-custom-nodes/releases/tag/v0.3.0


We use http for transferring blender to server,
<s>since blender could not import relative path modules.</s>
since things that are not thread safe is not allowed in blender plugin, and violates the principle of blender.

## Compositing Setup on Cycles

![](setup_cycles.png)

## Compositing Setup on EEVEE

![](setup_eevee.png)
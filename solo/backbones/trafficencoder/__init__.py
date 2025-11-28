# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from .trafficencoder import traffic_encoder as default_traffic_encoder
from .trafficencoder import traffic_encoder_simple as default_traffic_encoder_simple
from .trafficencoder import traffic_transformer as default_traffic_transformer
# from .trafficMoEEncode import traffic_moe_encoder as default_traffic_moe_encoder
def traffic_encoder_simple(method, *args, **kwargs):
    return default_traffic_encoder_simple(*args, **kwargs)
def traffic_encoder(method, *args, **kwargs):
    return default_traffic_encoder(*args, **kwargs)
def traffic_transformer(method, *args, **kwargs):
    return default_traffic_transformer(*args, **kwargs)
# def traffic_moe(method, *args, **kwargs):
#     return default_traffic_moe_encoder(*args, **kwargs)

__all__ = [
    "traffic_encoder_simple",
    "default_traffic_encoder",
    "traffic_transformer"
    # "traffic_moe"
    ]

# bytetrack

bytetrack追踪算法，仅包含追踪部分，接入检测模型推理数据即可使用

## Instructions

本代码只包含ByteTrack算法追踪部分。

因ByteTrack追踪算法可以仅依赖Detection模型即可实现高效率的追踪，因此我将ByteTrack追踪部分单独抽离出来，其接口可接入自己的检测模型结果。

## Experiment result

CPU: i7-8700@3.20GHz

Memory: 16G

GPU: None

Average person number per frame: 12

Algorithm average speed: 6.5ms

## References

The source code link [ifzhang/ByteTrack: ByteTrack: Multi-Object Tracking by Associating Every Detection Box (github.com)](https://github.com/ifzhang/ByteTrack)

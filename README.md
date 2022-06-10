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

## Warning

Bytetrack 关键算法部分使用了原作者的代码，但原作者的代码中将tlwh的list中的顺序写错了，但源代码逻辑却能自洽，能够正常运行，类似于C语言在开始的部分写入了#define TRUE FALSE。不要试图去修改这一bug，否则出现任何问题不要联系我。

## References

The source code link [ifzhang/ByteTrack: ByteTrack: Multi-Object Tracking by Associating Every Detection Box (github.com)](https://github.com/ifzhang/ByteTrack)

## Contact

QQ 744483644

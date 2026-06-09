# IQA-ODPS-UDTF 开源标准化 & 模型适配执行计划

## Phase 1: 代码整合与清理

- [x] 1.1 删除遗留包 `src/main/java/com/autonavi/liqe/` 全部文件
- [x] 1.2 更新 `pom.xml`：修改 mainClass、添加 JUnit 5 + Mockito 依赖、添加 surefire 插件
- [x] 1.3 创建 `src/main/java/com/autonavi/iqa/utils/TextFeatureLoader.java`（从遗留包迁移）
- [x] 1.4 修复 `LIQEModel.java` 的 import 和 `LIQEModelManager.java` 的接口实现
- [x] 1.5 移动 ODPS stubs：`src/main/java/com/aliyun/odps/` → `src/test/java/com/aliyun/odps/`
- [x] 1.6 创建统一 UDTF 入口 `ImageQualityUDTF.java`
- [x] 1.7 `ImagePreprocessor.java` 添加 `uniformCrop` 方法
- [x] 1.8 更新 `ModelRegistry.java` 注册全部 7 个模型

## Phase 2: ONNX 转换脚本（Python）

- [x] 2.1 `scripts/dbcnn_torch2onnx.py` — VGG16+SCNN+双线性池化，输入 (1,3,H,W)
- [x] 2.2 `scripts/hyperiqa_torch2onnx.py` — forward_patch 导出，输入 (1,3,224,224)
- [x] 2.3 `scripts/maniqa_torch2onnx.py` — 移除 hooks 改显式层索引，输入 (1,3,224,224)
- [x] 2.4 `scripts/musiq_torch2onnx.py` — 简化为单尺度固定输入，输入 (1,3,224,224)
- [x] 2.5 `scripts/tres_torch2onnx.py` — 仅导出 eval 路径（无翻转/一致性损失），输入 (1,3,224,224)
- [x] 2.6 `scripts/clipiqa_torch2onnx.py` — 预编码文本特征，导出图像编码器+评分头，输入 (1,3,224,224)

## Phase 3: Java 模型适配器（6 个新模型）

- [x] 3.1 DBCNN: `DBCNNModel.java` + `DBCNNModelManager.java`
- [x] 3.2 HyperIQA: `HyperIQAModel.java` + `HyperIQAModelManager.java`
- [x] 3.3 MANIQA: `MANIQAModel.java` + `MANIQAModelManager.java`
- [x] 3.4 MUSIQ: `MUSIQModel.java` + `MUSIQModelManager.java`
- [x] 3.5 TReS: `TReSModel.java` + `TReSModelManager.java`
- [x] 3.6 CLIPIQA: `CLIPIQAModel.java` + `CLIPIQAModelManager.java`

## Phase 4: 测试

- [x] 4.1 添加 JUnit 5 + Mockito 依赖（Phase 1 已完成）
- [x] 4.2 单元测试：QualityScore、ModelConfig、ModelRegistry、ModelFactory、ImagePreprocessor（共 48 个测试用例）
- [ ] 4.3 集成测试：每个模型的端到端测试（需 ONNX 模型文件，待模型转换后执行）

## Phase 5: 文档

- [x] 5.1 重写 README.md（完整文档，含 7 模型表格、架构图、转换详情、UDTF 示例）
- [x] 5.2 更新 docs/todo.md 标记已完成阶段

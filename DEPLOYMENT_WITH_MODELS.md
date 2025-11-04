# LIQE UDTF ODPS 部署指南（模型文件打包在 JAR 中）

## 更新说明

现在模型文件已经打包到 JAR 文件中，部署更加简单！

## 优势

- ✅ **不需要单独上传模型文件**：所有模型文件都在 JAR 中
- ✅ **部署更简单**：只需要上传一个 JAR 文件
- ✅ **版本一致**：代码和模型版本绑定在一起，避免不匹配

## 完整操作步骤

### 第一步：打包 JAR 文件（包含模型文件）

```bash
cd java_codes
./package_jar.sh
```

脚本会自动：
1. 检查模型文件是否存在（`../weights/onnx/`）
2. 将模型文件打包到 JAR 根目录
3. 打包所有依赖

生成的 JAR 文件：
- `target/liqe-udtf-1.0.0-jar-with-dependencies.jar`（包含所有依赖和模型文件）

### 第二步：上传到 ODPS

#### 2.1 上传 JAR 文件（只需要上传这一个文件）

```sql
ADD JAR /path/to/target/liqe-udtf-1.0.0-jar-with-dependencies.jar AS liqe-udtf.jar;
```

**注意**：现在不需要单独上传模型文件了！

### 第三步：创建 UDTF 函数

```sql
CREATE FUNCTION udtf_liqe
AS 'com.autonavi.liqe.UDTFLIQE'
USING 'your_project.liqe-udtf.jar';
```

**注意**：现在 `USING` 子句只需要 JAR 文件，不需要模型文件了！

### 第四步：使用 UDTF

```sql
SELECT 
  url,
  quality_score
FROM (
  SELECT 
    udtf_liqe(array(image_url)) AS (url, quality_score)
  FROM your_table
) t
WHERE quality_score > 0
```

## 代码说明

代码已经自动支持从 JAR 资源加载模型：

1. **优先尝试文件路径**（用于本地测试）：
   - 通过系统属性 `liqe.clip_model.path` 等指定
   - 或从文件系统查找

2. **如果文件路径不存在，自动从 JAR 资源加载**：
   - 使用 `getClass().getResourceAsStream("/clip_model.onnx")`
   - 模型文件必须放在 JAR 根目录

## 模型文件加载逻辑

```java
// 1. 首先尝试从文件路径加载（用于测试）
if (clipModelPath != null && !clipModelPath.startsWith("classpath:")) {
    modelManager.loadClipModel(clipModelPath);
} else {
    // 2. 从 JAR 资源加载
    InputStream clipStream = getClass().getResourceAsStream("/clip_model.onnx");
    modelManager.loadClipModel(clipStream);
}
```

## 验证部署

部署后可以运行测试查询：

```sql
SELECT 
  url,
  quality_score
FROM (
  SELECT 
    udtf_liqe(array('https://example.com/test.jpg')) AS (url, quality_score)
) t
```

如果返回正常结果（quality_score 在 1.0-5.0 之间），说明部署成功。

## 故障排查

### 1. 模型文件找不到错误

如果出现模型文件找不到的错误：
- 检查 JAR 文件是否包含模型文件：`jar tf liqe-udtf.jar | grep "clip_model.onnx"`
- 确保模型文件在 JAR 根目录（不是子目录）

### 2. 重新打包

如果修改了模型文件，需要重新打包：

```bash
cd java_codes
./package_jar.sh
```

## 对比：打包 vs 不打包模型文件

| 方式 | 优点 | 缺点 |
|------|------|------|
| **打包到 JAR** | 部署简单，只需一个文件；版本一致 | JAR 文件较大（~157MB+模型大小） |
| **单独上传** | JAR 文件较小；可以单独更新模型 | 需要上传多个文件；可能版本不匹配 |

**推荐**：使用打包方式，更简单可靠。


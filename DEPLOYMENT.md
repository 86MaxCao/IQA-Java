# LIQE UDTF ODPS 部署指南

## 完整操作步骤

### 第一步：打包 JAR 文件

#### 方法 1：使用 Maven（推荐）

如果你已经安装了 Maven：

```bash
cd java_codes
./package_jar.sh
```

或者手动执行：

```bash
cd java_codes
mvn clean package
```

这会生成两个 JAR 文件：
- `target/liqe-udtf-1.0.0.jar` - 主 JAR（不包含依赖）
- `target/liqe-udtf-1.0.0-jar-with-dependencies.jar` - **Fat JAR（包含所有依赖，用于 ODPS）**

**ODPS 部署需要使用 `jar-with-dependencies.jar`**

#### 方法 2：使用打包脚本（无需 Maven）

如果没有 Maven，可以使用提供的脚本：

```bash
cd java_codes
./package_jar.sh
```

脚本会自动：
1. 检查是否已编译
2. 提取所有依赖 JAR
3. 创建包含所有内容的 Fat JAR

### 第二步：准备资源文件

确保以下文件已准备好：

```bash
weights/onnx/clip_model.onnx
weights/onnx/liqe_model.onnx
weights/onnx/text_features.json
```

如果还没有转换，先运行 Python 转换脚本：

```bash
cd generate_poi_obj_cycling_and_walking
python scripts/liqe_torch2onnx.py --output_dir weights/onnx
```

### 第三步：上传到 ODPS

#### 3.1 上传 JAR 文件

```sql
-- 上传 Fat JAR（包含所有依赖）
ADD JAR /path/to/target/liqe-udtf-1.0.0-jar-with-dependencies.jar AS liqe-udtf.jar;
```

#### 3.2 上传模型文件

```sql
-- 上传 ONNX 模型和文本特征
ADD FILE weights/onnx/clip_model.onnx AS clip_model.onnx;
ADD FILE weights/onnx/liqe_model.onnx AS liqe_model.onnx;
ADD FILE weights/onnx/text_features.json AS text_features.json;
```

### 第四步：创建 UDTF 函数

```sql
CREATE FUNCTION udtf_liqe
AS 'com.autonavi.liqe.UDTFLIQE'
USING 
  'your_project.liqe-udtf.jar',
  'your_project.clip_model.onnx',
  'your_project.liqe_model.onnx',
  'your_project.text_features.json';
```

**注意**：
- `your_project` 需要替换为你的 ODPS 项目名
- 函数名 `udtf_liqe` 可以自定义
- `com.autonavi.liqe.UDTFLIQE` 是 UDTF 类的完整路径，不能修改

### 第五步：使用 UDTF

#### 基本用法

```sql
SELECT 
  url,
  quality_score
FROM (
  SELECT 
    udtf_liqe(array(image_url)) AS (url, quality_score)
  FROM your_table
) t
WHERE quality_score > 0  -- 过滤掉错误情况 (-1.0)
```

#### 批量处理多个 URL

```sql
SELECT 
  url,
  quality_score
FROM (
  SELECT 
    udtf_liqe(array(url1, url2, url3)) AS (url, quality_score)
  FROM your_table
) t
```

#### 与表数据结合

```sql
SELECT 
  t.id,
  t.image_url,
  result.url,
  result.quality_score
FROM your_table t
LATERAL VIEW udtf_liqe(array(t.image_url)) result AS url, quality_score
WHERE result.quality_score > 0
```

## 输出说明

- **输入**：`array<string>` - 图像 URL 数组
- **输出**：
  - `url` (string) - 图像 URL
  - `quality_score` (double) - 质量分数
    - **正常范围**：1.0 - 5.0
    - **错误标识**：-1.0（处理失败）

## 重要提示

1. **输出数量保证**：UDTF 保证输出数量严格等于输入数量，即使处理失败也会返回 `-1.0`
2. **错误处理**：所有错误情况都会返回 `-1.0`，可以通过 `WHERE quality_score > 0` 过滤
3. **性能**：UDTF 内部使用批处理（每批 10 个 URL）提高效率
4. **资源路径**：确保 ODPS 能够访问上传的资源文件

## 故障排查

### 1. 类找不到错误

如果出现 `ClassNotFoundException`：
- 检查是否使用了 `jar-with-dependencies.jar`（不是普通的 jar）
- 检查 JAR 文件是否正确上传

### 2. 模型文件找不到

如果出现模型文件找不到的错误：
- 检查资源文件是否正确上传
- 检查 `USING` 子句中的资源名称是否正确
- 检查项目名称是否正确

### 3. 运行时错误

查看 ODPS 日志获取详细错误信息：
- 检查网络连接（下载图像）
- 检查图像格式是否支持
- 检查 ONNX 模型文件是否完整

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


# 安装 Maven 指南

## 选项 1: 使用 Homebrew (推荐，如果已安装)

```bash
brew install maven
```

## 选项 2: 手动安装 Maven

### macOS:

1. 下载 Maven:
```bash
cd ~/Downloads
wget https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz
tar -xzf apache-maven-3.9.6-bin.tar.gz
sudo mv apache-maven-3.9.6 /usr/local/maven
```

2. 配置环境变量 (添加到 ~/.zshrc):
```bash
export M2_HOME=/usr/local/maven
export PATH=$M2_HOME/bin:$PATH
```

3. 重新加载配置:
```bash
source ~/.zshrc
```

4. 验证安装:
```bash
mvn -version
```

## 选项 3: 使用 Maven Wrapper (无需全局安装)

项目可以包含 Maven Wrapper，这样就不需要全局安装 Maven。

## 选项 4: 手动编译 (不使用 Maven)

我已经创建了 `compile_and_test.sh` 脚本，可以手动下载依赖并编译。

**注意**: 最简单的方式是安装 Maven，然后直接运行 `mvn clean compile`。


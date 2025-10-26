# プロジェクト実行方法

## 手順

1. リポジトリをクローン
```
   git clone https://github.com/yourname/project.git
   cd project
```
2. Docker イメージをビルド
```
   docker build -t langchain-1 .
```
3. .env ファイルを作成
```
   cp .env.example .env
```
4. 実行
```
   docker run -it --rm --env-file .env langchain-1
```

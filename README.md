# プロジェクト実行方法

## 手順

1. リポジトリをクローン
   git clone https://github.com/yourname/project.git
   cd project

markdown
コードをコピーする

2. Docker イメージをビルド
   docker build -t langchain-1 .

bash
コードをコピーする

3. .env ファイルを作成
   cp .env.example .env

API キーなどを入力
markdown
コードをコピーする

4. 実行
   docker run -it --rm --env-file .env langchain-1

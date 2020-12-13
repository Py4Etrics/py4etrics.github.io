# ツールのインストールと説明

## 説明

Pythonのインストールと関連ツールについて簡単な説明をする。
* Atomのインストール
* Pythonのインストール
* Gitのインストール
* Terminal（Mac）とGit Bash（Windows）の使い方
* Gitのインストールの確認と設定
* Pythonのインストールの確認と設定
* GitHubの設定
* pipを使うパッケージのインストール
* Jupyter Notebookの使い方

また次の点についても述べる。
* ゼミ終了後毎回おこなう作業

```{note}
* `Terminal`と`Git Bash`ではコマンドを入力する行に`$`が表示される。
* 「コマンド入力」とは`$`の後にコマンドをタイプすること。
* 「実行」とはEnter/Returnキーを押すこと。
```

## Atomのインストール
Atomはテキスト・ファイルを編集するText Editorと呼ばれる有名なアプリの１つである。授業での活躍の場は多くないが，今のうちに慣れるのも良いだろう。

[このリンクから](https://atom.io)からダウンロードできる。

## Pythonのインストール
AnacondaとはPythonと広く使われるパッケージを一括でダウンロード・インストールできるプラットフォームである。（パッケージについてPythonの基礎で説明する。）

[このサイト](https://www.anaconda.com/distribution/#download-section:)からMac用もしくはWindows用のインストーラーをダウンロードする。必ず**Python 3.x version**を選ぶこと。Anacondaのインストーラーの指示に従いデフォルトでインストールする。

## Gitのインストール
GitとはPC内でコードの履歴管理をおこなうことによりファイルの過去のバージョンの閲覧や復元などを可能にするアプリ。特に，以下で説明するGitHubと一緒に使うことにより，複数人と共同作業する場合に欠かせないアプリである。以下ではMacとWindowsを別々に説明する。

### Mac
- Macには元々インストールされているので再インストールする必要はない。
- コマンドライン・デベロッパ・ツールについて
    - 次のメッセージのポップアップ・ウィンドウが表示される場合，「インストール」をクリックすること。
        ```
       gitコマンドを実行するには，コマンドライン・デベロッパ・ツールが必要です。ツールを今すぐインストールしますか？"　「インストール」ボタンをクリック
        ```
    - 以下で説明するTerminalで次のコマンドを実行してインストールすることもできる。
        ```
        $ xcode-select —-install
        ```

### Windows
- [このサイト](https://git-scm.com)から`Git`をダウンロードする。
- ダウンロードしたファイルを起動しインストールする。
    - `Next`を押していくと様々なオプションが出てくるが次の項目以外はすべてデフォルトを選択する。
    - `Choosing the default editor used by Git`の画面が出てくるので，そのリストの中に使うEditorがあれば（例えば、Atomを選択し、Atomをインストールしていなければ`Use the Nano editor by default`を選択する。

## Terminal（Mac）と Git Bash（Windows）

* MacにはTerminalがもともとインストールされている。
* WindowsにGitをインストールする際，Git Bashは一緒に自動でインストールされる。
* TerminalとGit Bashは，コマンド（例えば，`ls`）を入力し，そのコマンドを実行することによりPCを制御する。これらを使う理由はCLI（Command Line Interface）を通してGitとGitHubを使うためである。
* WindowsではAnaconda Promptも使うことができるが，ここではGit Bashだけを取り上げる。
* Git Bashは，WindowsでもMacのTerminalと同じCLIが使えるように開発されたものなので，コマンド（次の「３種のコマンド」も含めて）は共通である。

### 「３種のコマンド」
以下では必須の3つのコマンドについて説明する。

#### pwd
ハードディスクにあるファイルは階層構造になっているフォルダー（ディレクトリとも呼ばれる）に保存されている。Terminal自体が「位置する」フォルダーをカレント・ディレクトリ（Current Directory）と呼び，それを表示するためのコマンドがpwd（Print Working Directoryの略）である。
```
$ pwd
```
を実行するとカレント・ディレクトリが表示される。例えば、
```
/Users/myName/folder1/folder2
```
この例では，カレント・ディレクトリは`folder2`となっている。

#### ls
カレント・ディレクトリにあるファイルやフォルダーを表示するコマンド。`ls`（listの略）の後にオプションを付けることもできる。
```
$ ls      # ファイルやフォルダーの表示
$ ls -a  # 隠しファイルと隠しフォルダーも含めて表示
```
ここで隠しファイルと隠しフォルダーとは、名前が`.`で始まるものであり，通常は見えなくなっている。

```{note}
Macの`Finder`で `Command + Shift + .`とすると隠しファイルと隠しフォルダーの表示・非表示をトグルできる。
```

#### cd
`cd`（Change Directoryの略）は，カレント・ディレクトリーを他のフォルダーに移す場合に使うコマンド。ここでの説明は「相対パス」を使う。基本的な使い方は
```
$ cd ＜移動したいフォルダー名＞
```
の形である。上の例を考えよう。
* 下の階層フォルダーに移動する場合
    * カレント・ディレクトリーが`folder2`であり，その下層フォルダーである`folder3`に移る場合（`folder3`は`folder2`に含まれるフォルダー）
```
$ cd folder3
```
* 上の階層フォルダーに移動する場合
    * `folder3`に移った後，`folder2`に戻る場合
```
$ cd ..
```
ここで上層フォルダーは１つしかないのでフォルダー名を入れる代わりに`..`で事足りるということである。

```{note}
* ファイル名やフォルダー名を途中まで書いてタブを押すと補完してくれる。
* 以下のコマンドを実行するとホーム・ディレクトリー（自分の名前のディレクトリー）に移動できる。
    ```
    $ cd ~
    ```
    ここで`~`はホーム・ディレクトリーを表す。
```


## Gitインストールの確認と設定
### Mac
#### 確認
MacにはGitはもともとインストールされていると説明したが，確認するためにTerminalで
```
$ which git
```
を実行すると`Git`があるフォルダーが表示される。

#### 設定
GitにTerminalを使って名前とメールアドレスを登録するが，以下で説明するGitHubで登録するものと同じでなくてはならない。
```
$ git config --global user.name "LastName_FirstName"
$ git config --global user.email "example@example.com"
```
これ以外に必要であれば
```
$ git config --global core.editor 'atom'    # エディターの設定
$ git config --global --list` 　　　　　　　# 設定の確認
```

### Windows
#### インストールの確認
Git Bashの画面で以下を実行する（どのカレント・ディレクトリでも構わない）。
```
$ which git
```
以下のように表示されればインストール成功。
```
/cmd/git
```

#### 設定
GitにGit Bashを使って名前とメールアドレスを登録するが，以下で説明するGitHubで登録するものと同じでなくてはならない。
```
$ git config --global user.name "LastName_FirstName"
$ git config --global user.email "example@example.com"
```
これ以外に必要であれば，
```
$ git config --global --list` 　　　　　　　# 設定の確認
```

## Pythonインストールの確認
### Mac
Terminalの画面で以下を実行しよう（どのカレント・ディレクトリでも構わない）。
```
$ which python
```
以下（パスに`anaconda3`が入っている）のようにされれば成功。
```
/Users/myName/anaconda3/bin/python
```
もし
```
/usr/bin/python
```
のように`anaconda3`がパスに入っていない場合は，
```
$ conda activate
$ which python
```
を実行してみること。ここで
```
$ conda activate
```
はAnacondaを使えるようにするコマンドであり，その逆は
```
$ conda deactivate
```
である。
```{note}
`conda activate`及び`conda deactivate`は、一度設定すれば終わりというものではなく必要であれば適宜実行する必要があるコマンドである。
```

### Windows
* anaconda3がインストールされたディレクトリを確認する。２つのケースを考える。
    ```
    （ケース１）C:\Users\myName\Anaconda3
    （ケース２）C:\Anaconda3
    ```
* Git Bashで以下を実行する。
    * （ケース１）の場合
    ```
    echo ". /c/Users/myName/Anaconda3/etc/profile.d/conda.sh" >> ~/.profile
    ```
    * （ケース２）の場合
    ```
    echo ". /c/Anaconda3/etc/profile.d/conda.sh" >> ~/.profile
    ```
    * 両方のケースに当てはまらない場合は，`/c/`と`/etc/`の間を適宜変更する。
    * この作業によりホーム・ディレクトリに`.profile`というファイルが作成され，Pythonに「パスがとおる」（即ち、Git Bashのカレント・ディレクトリがどこであれ使うことができる）ことになる。
* Git Bashを再起動する。これによりGit Bashは`.profile`の設定ファイルを読み込んむことになる。
* Git Bashで以下を実行する（どのカレント・ディレクトリでも構わない）。
```
$ conda activate
$ which python
```
を実行してみること。ここで
```
$ conda activate
```
はAnacondaをアクティベート（使えるように）するコマンドであり，その逆は
```
$ conda deactivate
```
である。以下のように表示されれたパスにanaconda3が入っていれば成功。
```
/c/Users/myName/Anaconda3/python
```

## GitHubの設定
GitHubとは，コードとコードの履歴をクラウド上で管理し，コード作成の共同作業を手助けするアプリである。アップロードしたコードを公開・非公開に設定することができる。ゼミ生がアップロードしたコードは外部からは春山のみが閲覧可能となる。

* **GitHubの設定**
    * Gitの設定で使ったメールアドレスを使い[このサイト](https://github.com)で無料アカウントを作成する。
    * 登録後，[このサイト](https://education.github.com/students)でStudent Accountに変更する。
        * このでは Add an email address と書かれたボタンを押して神戸大学の学番メールアドレスを入力すること。
        * 学番アドレスに確認メールが送られてくるので，そのメールに従って作業する。
        * Student Accountにすると様々な[特典](https://education.github.com/pack#offers)が付いてくる。
    * GitHubの画面の左側に”レポジトリ”（repository）を作成するボタンがあるので，それを押して作成する
        * レポとはコードをアップロードした際に保存される領域を示す。
        * レポジトリを「レポ（repo）」と略す場合がよくある。
        * それをゼミようのレポジトリとする。
        * レポは必要な数だけ作成することができる。(ゼミでは１つのレポを使う)
    * 春山に`username`のみを知らせる。
    * Passwordは後で使うので忘れないように。
* **GitとGitHubの同期の設定：Part I**
    * ゼミだけで使い，GitHubと同期させるフォルダーを作成する。
        * 以下ではそのフォルダーを`ZF`（ゼミ・フォルダー）と呼ぶ。
        * Macでは「書類」フォルダー，Windowsでは「ドキュメント」フォルダーに`ZF`を作成することを推奨する。
        * (注意：Mac) iCloud Drive用のフォルダーを指定しないように。（Finder → 移動 → コンピュータ → ...）
    * TerminalもしくはGit Bashを起動し，カレント・ディレクトリを`ZF`にする。
    * `ZF`とGitHubを連携させるためにTerminalもしくはGit Bashで以下を実行する。
        * `ZF`をGit用に設定する。
          ```
          $ git init
          ```
        * `ZF`とGitHubを紐付ける
          ```
          $ git remote add origin <repoURL>
          ```
        ここで`<repoURL>`は上で作成したレポのアドレスであり，以下の手順でコピーできる。
        * repoのページにある`Clone or download`と書かれた緑色のボタンをクリックし，`https://`から始まるリンクを確認する。その右横のボタンを押す。
        * リンクが`git@github.com:`から始まっている場合は，その小さなポップアップ・ウィンドウの右上にある`Use HTTPS`と書かれた箇所をクリックすると`https://`から始まるリンクが表示されるので，それをコピーする。
* **GitとGitHubの同期の設定：Part II**
    * エディター（例えば，MacであればAtomやテキストエディット，WindowsであればAtomやメモ帳）を使い`README.md`というファイルを`ZF`に作成し，自己紹介の内容を記入する（例えば，自分の名前や学部など）。
    * 以下はすべてカレント・ディレクトリを`ZF`にしたTerminalもしくはGit Bashでおこなう。
        1. `ZF`の状態を確認する。
            ```
            git status
            ```
        2. Gitに履歴を記録させるファイル（この場合は`README.md`）を選択する。これは「ステージング」と呼ばれるプロセス。
            ```
            git stage README.md
            ```
            もしくは
            ```
            git add README.md
            ```
        3. Gitにファイルの変更を登録する。これは「コミット」と呼ばれるプロセス。
            ```
            git commit -m "First commit"
            ```
            ここで`-m`はその後にメッセージを入れるオプション。「First Commit」は履歴についての短いコメントであり，毎回記述する必要がある。変更内容について分かりやすいものすること。もしくは，
            ```
            git commit
            ```
            としても良いが，その場合，Gitを設定した時に指定したエディターが起動されるので，「First commit」のようなコメントを書いて保存する。エディターは終了しても良い。
        4. GitHubにファイルをアップロードする。これは「プッシュ」と呼ばれるプロセス。
            ```
            git push origin master
            ```
        5. パスワードを求められるので，GitHubのパスワードを入力する。
        6. `ZF`の状態を確認する。
            ```
            git status
            ```
    * 基本的にゼミ終了後毎回１〜６のステップ（README.mdを違うファイルに変えて）をおこなうことになる。
* **ゼミのための設定**
    * ゼミ用レポのGitHubウェブページを開く。README.mdの内容が表示されているはず
    * 以下の設定をおこない，春山だけがそれぞれの学生のレポを閲覧することが可能となる。
        * Settings -> Manage access を開き， Invite a collaborator と書かれた緑色のボタンを押す。
        * 出てきたポップアップに春山のユーザー名`spring-haru`を入力すると春山の名前が出てくるのでマウスで選択する。
        * Add spring-haru to ... と書かれた緑色のボタンをおす。

```{warning}
コンフリクトと呼ばれる問題発生を防ぐためにGitHubのサイトで直接ファイルを変更**しないように！**もし変更した場合は，パソコンで該当ファイルを変更する**前に必ず**以下のコードを実行すること。

    $ git pull origin master
```

## pipを使うパッケージのインストール

ゼミではAnacondaに含まれていない次のパッケージを使う。
* `linearmodels`
* `wooldridge`
* `lmdiag`
* `see`

これらをインストールするには，Anacondaに含まれている`pip`コマンドを使う。`pip`は以Terminal（Mac用）もしくはGit Bash（Windows用）を使い実行する。例えば，`see`をインストールするには
```
$ pip install see
```
を実行する。4つのパッケージを一括でインストールするコマンドは次のようにする。
```
$ pip install linearmodels wooldridge lmdiag see
```

## Jupyter Notebookの使い方
Jupyter Notebookとは，Pythonをインターアクティブな環境で実行できる非常に使い易いプログラムである。

### 起動と終了（Mac）
* 起動（２つの方法）
    1. Terminalに
        ```
        Jupyter Notebook
        ```
        を入力し実行する。

    1. Anaconda-Navigatorを起動するとJupyter Notebookの大きなアイコンがあり，そこにある`Launch`をクリックする。
* 終了
    * Jupyter Notebookの全てのタブを閉じる。
    * Jupyter Notebookを起動すると自動にTerminalの画面が表示されることになるが，次のどれかの方法で，その画面に`$`が表示されるようにする。
        * `Control`を押したまま`C`を押し，`y`を入力し，`Enter`を押す。
        * `Control`を押したまま`C`を２回連続で押す。
        * 別のTerminal画面の`$`で`jupyter notebook stop`を実行する。

### 起動と終了（Windows）
* 起動（３つの方法）
    1. Git Bashに
        ```
        Jupyter Notebook
        ```
        を入力し実行する。
    1. スタートメニューから`Jupyter Notebook (Anaconda3)`をクリック。
    1. Anaconda-Navigatorを起動するとJupyter Notebookの大きなアイコンがあり，そこにある`Launch`をクリックする。
* 終了
    * Jupyter Notebookの全てのタブを閉じる。
    * Jupyter Notebookを起動すると自動でAnaconda Promptと呼ばれるアプリが起動する。その画面に移り，`Control`を押したまま`C`を押すと画面が閉じる。もしくは`C:¥Users¥myName>`のような表示がでる。（代替方法として，別のAnaconda Promptを起動して`jupyter notebook stop`を実行しても良い。）

### 使い方
* ブラウザーが起動した後，Jupyter Notebookの最初の画面ではファイルやフォルダーが表示される。
    * フォルダー名をクリックすると，そのフォルダーに移動する。
    * `..`をクリックすると，１階層上のフォルダーに移動する。
* `FZ`に移動し，以下の手順で新しいNotebookを作成する
    * 画面右上にある`New`をクリックして表示される`Python 3`をクリックする。
    * 画面左上に表示される`Untitled`とクリックしファイル名を変更する。
* Notebookにある横長の長方形をセルと呼ぶ。主に２種類のセルを使う。
    * Codeセル
        * セル枠の左側に`ln[]:`の表示がある。
        * `Python`コードを書く領域
        * `Shift+Enter`でコードが実行されCodeセルの下に結果が表示される
    * Markdownセル
        * セル枠の左側に何の表示もない。
        * 説明文などを書く領域
        * `Shift+Enter`で内容が表示される
        * [Markdownの書き方の参考サイト](https://learnxinyminutes.com/docs/markdown/)
    * セル・タイプの変更は，画面中央上に`Code`または`Markdown`と表示されている部分をクリックして該当するセルタイプを選択する。
* Jupyter Notebookの画面の状態には２つのモードがある。
    * エディット・モード（EM）
        * セル枠の色が緑色
        * セル内にカーソルがある。
        * セルにコードや説明文を記入できる。
    * コマンド・モード（CM）
        * セル枠の色が青色
        * セル内にカーソルがない。
        * 矢印キーを使ってセル間を移動できる。（マウス・クリックでも移動できる。）
    * EMとCM間の移行方法
        * EMにある状態で`Shift+Enter`で（セルが実行され）CMに移行する。
        * EMにある状態で`Escape`を押すとCMに移行する。
        * CMにある状態で`Enter`を押すとEMに移行する（セル内にカーソルが現れる）。
* コード・セル内でコマンドの末尾に`?`と書いて実行すると、そのコマンドの`docstrings`と呼ばれる説明文（英語）が表示される。`help(<command>)`でも同じ（ここで`<command>`とはPythonの何らかのコマンド）
* コマンドを途中まで入力して`Tab`を押すと残りのコマンドを補完してくれる。複数表示される場合は、必要なものを１つを選択すること。選択したくない場合は`Escape`を押すと消える。
* 関数や`()`の中で`Shift+Tab`を押すとツールチップ（tooltips）（即ち、ヒント）を表示してくれる。

### Tips
* メニューのHelp -> Edit Keyboard Shortcutsからショートカットを変更・設定できる。
* [Jupyter Notebook Extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)は便利。
* コードをより見やすくできるフォントもあるのでトライしてみよう。
* `matplotlib`を使う場合の日本語
  * [参考サイト](https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html)
  * Macユーザーは[このリンク](https://raw.githubusercontent.com/Haruyama-KobeU/Haruyama-KobeU.github.io/master/data/matplotlibrc)のファイルをダウンロードして`/Users/ユーザー名前/.matplotlib/matplotlibrc`として保存すると図に日本語が表示される。
* Web上にはJupyter Notebookの使い方に関する情報がたくさんあるのでチェックしてみよう。
  * [Google検索結果](https://www.google.co.jp/search?q=jupyter+notebook+%E4%BD%BF%E3%81%84%E6%96%B9)
  * [キーボード・ショートカット](https://qiita.com/forusufia/items/bea3f6fd6160cd2f5843)

## ゼミ終了後毎回おこなう作業
以下の手順で，ゼミ終了後には毎回必ず使ったJupyter NotebookをGitHubにアップロードすること。
1. TerminalもしくはGit Bashを使い`ZF`をカレント・ディレクトリにする。
1. `ZF`のどのファイルが変更されてGitHubと同期されていないかをチェックする。
    ```
    git status
    ```
1. 変更ファイルのステージング（変更履歴の保存をするファイルの選択）
    ```
    git stage file名
    ```
    ここで`stage`を`add`としても同じ。
1. ファイルのコミット（変更履歴の保存）
    ```
    git commit -m '変更点に関するコメント'
    ```
1. プッシュ（GitHubにプッシュ）
    ```
    git push origin master
    ```
1. `ZF`の状況を確認
    ```
    git status
    ```
1. ブラウザーでGitHubのレポをチェック

```{warning}
コンフリクトの問題発生を防ぐためにGitHubのサイトで直接ファイルを変更**しないように！**
もし変更した場合は，パソコンで該当ファイルを変更する**前に必ず**以下のコードを実行すること。

    $ git pull origin master

`pull`はGitHub上のファイルをPCにダウンロードしてPCのレポを上書きする。これによりリモート・レポ（GitHub上のレポ）とローカル・レポ（PCのレポ）を同期することになる。
```

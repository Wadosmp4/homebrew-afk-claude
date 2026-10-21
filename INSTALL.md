# Getting started with AFK Claude

AFK Claude lets you watch your Claude Code and Codex sessions from your phone, answer their questions, approve or deny the commands they want to run, and send them new instructions.

It has two parts you install:

| Part | Where it runs | What it does |
|---|---|---|
| **Companion** (this package) | Your Mac or Linux computer | Starts and watches the sessions |
| **App** | Your phone | Where you read, approve and reply |

Your code stays on your computer. Only the session messages (what Claude says, the commands it asks to run, your replies) pass through our service on their way between the two.

## What you need

- A **Mac** or a **Linux** computer (Ubuntu, Debian, Fedora and similar). Windows works only through WSL2 - see [Windows](#windows).
- **Claude Code installed and signed in on that computer.** Install it from the official Claude Code instructions, then run `claude` once in a terminal and log in. The companion drives that installed copy and uses its login. (The `pip` install below bundles its own copy; the Homebrew one does not.)
- **Homebrew** (`brew.sh`). It is the easiest way to install and to keep the companion updated. No Homebrew? See [Without Homebrew](#without-homebrew).
- The **AFK Claude app** on your phone, and the **invitation code** you were sent (one long string starting with `AFK1.`).

## 1. Install the companion

```bash
brew tap Wadosmp4/afk-claude
brew install afk-claude-companion
```

Check it worked:

```bash
afk-claude-companion --help
```

## 2. Connect the computer

Run this:

```bash
afk-claude-companion setup
```

It asks whether you have an invitation code: answer `y` and paste the invitation code you were sent. (It is one long string starting with `AFK1.`)

It then prints a **6-digit code**. Keep that window open for the next step. The code works once and expires after a few minutes. If it runs out, run `afk-claude-companion setup` again for a fresh one.

## 3. Install the app and pair your phone

1. Install the AFK Claude app using the link you were sent (see [Installing the app](#installing-the-app)).
2. Open it. On the first screen, enter the 6-digit code from step 2.
3. You should land on **Your computers**, with your computer listed.

## 4. Keep the companion running

The companion has to be running for your phone to see the computer. Start it now and have it start by itself at every login:

```bash
brew services start afk-claude-companion
```

(Prefer to run it by hand? `afk-claude-companion run` in a terminal. It stops when you close the terminal.)

Your computer should now show **Online** in the app. Tap **View sessions**, pick a project and start a session.

## Using Codex too (optional)

Claude Code works out of the box. Codex sessions are optional and need an extra component that the Homebrew install does **not** include yet. To use Codex, install the companion with `pip` instead and ask for the Codex extra:

```bash
python3 -m venv ~/.afk-claude && ~/.afk-claude/bin/pip install "remote-claude-companion[codex] @ git+https://github.com/Wadosmp4/homebrew-afk-claude.git"
```

(See [Without Homebrew](#without-homebrew) for running it.) Then sign in to Codex on that computer, or add an OpenAI key under **Settings → Codex → My key** in the app. If you only use Claude, skip this: picking Codex in the app will say it couldn't start.

## Adding another computer

On the new computer, install the companion (step 1), then run:

```bash
afk-claude-companion setup
```

In the app, open **Home → Add a computer** and copy the connect command it shows, then run it on the new computer. It prints a 6-digit code. Back in the app, tap **Enter the 6-digit code** and type it. You can give the computer a name there.

## Installing the app

| Phone | How |
|---|---|
| **Android** | Open the install link you were sent on the phone. Android will ask to allow installs from your browser - allow it for this one install. |
| **iPhone** | Open the TestFlight invitation you were sent, install TestFlight if asked, then install AFK Claude from it. |

The app is a test version. Updates arrive the same way: a new link (Android) or an update in TestFlight (iPhone).

## Updating the companion

```bash
brew update
brew upgrade afk-claude-companion
brew services restart afk-claude-companion
```

## Something isn't working

| What you see | What to do |
|---|---|
| The app says your computer is **Offline** | Is the companion running? `brew services list` should show `afk-claude-companion started`. If not: `brew services start afk-claude-companion`. If it says started and the app still says Offline, run `brew services restart afk-claude-companion`, wait a minute, then pull down on the app's Home screen. |
| "That code is incorrect" when pairing the **phone** | The code expired or was already used. Run `afk-claude-companion pairing-code` on the computer for a new one. |
| "That code is incorrect, already used, or unknown" in **Settings → Pair a new computer** | That screen only takes a code from `afk-claude-companion setup` on a *different* computer that is waiting to join. To pair a phone, use the first-launch screen instead. |
| Sessions start but Claude answers with a login error, or nothing starts | Make sure `claude --version` works in a terminal on that computer, then run `claude` and sign in again. |
| Opening a project from the app does nothing, or file editing never loads | The companion is out of date. Update it (see above). |
| Everything looks stuck | `brew services restart afk-claude-companion`, then close and reopen the app. |

Still stuck? Send whoever invited you the output of `brew services list` and `afk-claude-companion --help`, and what the app shows.

## Windows

The companion does not run natively on Windows. If you use **WSL2** (Windows Subsystem for Linux, with Ubuntu), install Homebrew inside it and follow the steps above - it behaves like a Linux computer. The projects you want to work on need to live inside WSL. This route has not been tested by the project, so expect some rough edges.

## Without Homebrew

If you can't use Homebrew, you can install the same package with Python 3.10 or newer:

```bash
python3 -m venv ~/.afk-claude && ~/.afk-claude/bin/pip install "git+https://github.com/Wadosmp4/homebrew-afk-claude.git"
~/.afk-claude/bin/afk-claude-companion setup
~/.afk-claude/bin/afk-claude-companion run
```

You then have to start it yourself after a restart (or add it to your own service manager).

## Your privacy and your control

- The companion only starts and watches sessions on your computer. The app can only act through it.
- Each phone and each computer has its own key. In the app, **Settings → Paired devices → Revoke** cuts one off at once, for example a lost phone.
- Session messages pass through our service between your computer and your phone, so don't put anything in a session you wouldn't want it to carry.
- Commands that Claude wants to run wait for your approval in the app, unless you chose a more automatic permission mode for a session.

## Removing it

```bash
brew services stop afk-claude-companion
brew uninstall afk-claude-companion
brew untap Wadosmp4/afk-claude
rm -rf ~/.config/remote-claude-companion
```

Then revoke the computer in the app under **Settings → Paired devices**.

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
# terok:container — concatenated into the system bashrc; never executed standalone.

# Help banner (interactive login only)
if [ -t 1 ] && [ -z "$_TEROK_BANNER_SHOWN" ]; then
  export _TEROK_BANNER_SHOWN=1
  _TEROK_LOGIN=1 hilfe --kurz 2>/dev/null || true
  _terok_agent_history="$HOME/.terok/agent-history"
  if [ -s "$_terok_agent_history" ]; then
    awk '
      NF && !seen[$0]++ {
        printf "%s%s", count++ ? ", " : "Agents previously active: ", $0
      }
      END { if (count) printf "\n" }
    ' "$_terok_agent_history" 2>/dev/null || true
  fi
  unset _terok_agent_history
  if [ -s /home/dev/.terok/initial-prompt.txt ]; then
    printf '\033[1;33m📝 Initial prompt\033[0m \033[2m(/home/dev/.terok/initial-prompt.txt)\033[0m\n'
    cat /home/dev/.terok/initial-prompt.txt
    printf '\n\n'
  fi
fi

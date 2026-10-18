# MAG settings for running the agents on NLR Kestrel (or any machine without ALCF's proxies).
# aurora_env_setup.sh sets ALCF proxy variables that break network access elsewhere.
# Put your token below, and don't commit it.
#
#   source kestrel_env_setup.sh
#   python scaling_agents/check_mag_agent.py

export AMSC_TOKEN="<INSERT YOUR TOKEN>"
export ANTHROPIC_AUTH_TOKEN=$AMSC_TOKEN
export ANTHROPIC_BASE_URL=https://mag.genesis.americansciencecloud.org
export ANTHROPIC_CUSTOM_HEADERS="X-AmSC-MAG-Provider: google"
export GINO_LLM_MODEL=claude-opus-5-5

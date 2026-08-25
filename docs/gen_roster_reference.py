# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Generate the roster reference and JSON schemas from Pydantic models.

The mkdocs-gen-files plugin runs this script during ``mkdocs build``. The script
reads these contracts:

- [`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml] for agent YAML
- [`RawProvider`][terok_executor.roster.schema.RawProvider] for provider YAML
- [`VaultRouteEntry`][terok_executor.roster.schema.VaultRouteEntry] for ``routes.json``

The script creates these files:

- A Markdown roster reference with field tables and an agent example
- ``schemas/agent.schema.json`` for agent-YAML completion
- ``schemas/provider.schema.json`` for provider-YAML completion
- ``schemas/routes.schema.json`` for validation of generated vault routes

Each Pydantic ``Field`` description is the only source for its field description.
"""

from __future__ import annotations

import io
import json

import mkdocs_gen_files
from mkdocs_terok.config_reference import (
    render_json_schema,
    render_model_tables,
    render_yaml_example,
)
from pydantic import TypeAdapter

from terok_executor.roster.schema import RawAgentYaml, RawProvider, VaultRouteEntry

_MD_RULE = "---\n\n"


def _generate() -> str:
    """Return the complete ``roster-reference.md`` content."""
    buf = io.StringIO()
    buf.write("# Agent and Provider Roster Reference\n\n")
    buf.write(
        "This page is **auto-generated** from the Pydantic schema in "
        "[`roster.schema`][terok_executor.roster.schema]. Terok validates every "
        "listed field when it loads a file. Terok rejects unknown keys. This "
        "behavior identifies typing errors before Terok uses default values.\n\n"
    )
    buf.write(
        "**JSON Schema files for editor completion and validation:**\n\n"
        "[:material-download: agent.schema.json](schemas/agent.schema.json){: .md-button }\n"
        "[:material-download: provider.schema.json](schemas/provider.schema.json){: .md-button }\n"
        "[:material-download: routes.schema.json](schemas/routes.schema.json){: .md-button }\n\n"
    )

    buf.write(_MD_RULE)
    buf.write("## Agent YAML\n\n")
    buf.write(
        "Terok parses each bundled file in ``resources/agents/*.yaml``. It also "
        "parses each user override in ``~/.config/terok/agent/agents/*.yaml``. "
        "Each file becomes a "
        "[`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml] object. Terok "
        "then converts the object to a type in "
        "[`roster.types`][terok_executor.roster.types].\n\n"
        'All sections use ``extra="forbid"``. Thus, an unknown field such as '
        "``headles:`` or ``prommpt_flag:`` causes an error. Terok does not use a "
        "default value for an unknown field.\n\n"
    )
    buf.write(render_model_tables(RawAgentYaml))

    buf.write("### Full example\n\n")
    buf.write('```yaml title="claude.yaml"\n')
    buf.write(render_yaml_example(RawAgentYaml))
    buf.write("```\n\n")

    buf.write(_MD_RULE)
    buf.write("## Provider YAML\n\n")
    buf.write(
        "Terok parses each bundled file in ``resources/providers/*.yaml``. "
        "Terok also parses each user file in ``~/.config/terok/providers/*.yaml``. "
        "Each file becomes a "
        "[`RawProvider`][terok_executor.roster.schema.RawProvider] object.\n\n"
        "Terok loads the legacy ``~/.config/terok/agent/providers/*.yaml`` directory "
        "first. A file in the current provider directory overrides a legacy file that "
        "has the same name.\n\n"
        "The file name, without ``.yaml``, is the provider name. The name must match "
        "``[a-z0-9]+``. Thus, use only lowercase ASCII letters and digits. A new "
        "provider name must not match an existing agent or tool name.\n\n"
        "See [Custom providers](agents.md#custom-providers) for a minimal example. "
        "The example declares model data, so OpenCode and Pi do not request the "
        "``/models`` endpoint.\n\n"
    )
    buf.write(render_model_tables(RawProvider))

    buf.write(_MD_RULE)
    buf.write("## Generated routes.json\n\n")
    buf.write(
        "[`AgentRoster.generate_routes_json()`][terok_executor.roster.loader.AgentRoster.generate_routes_json] "
        "creates the ``routes.json`` file. The sandbox vault server reads this "
        "file. Each entry complies with "
        "[`VaultRouteEntry`][terok_executor.roster.schema.VaultRouteEntry]. The "
        "top-level object maps each provider name to its entry. Serialization omits "
        "empty optional fields.\n\n"
    )
    buf.write(render_model_tables(VaultRouteEntry))

    return buf.getvalue()


_routes_adapter: TypeAdapter[dict[str, VaultRouteEntry]] = TypeAdapter(dict[str, VaultRouteEntry])


with mkdocs_gen_files.open("roster-reference.md", "w") as f:
    f.write(_generate())

with mkdocs_gen_files.open("schemas/agent.schema.json", "w") as f:
    f.write(render_json_schema(RawAgentYaml, title="terok-executor agent YAML"))

with mkdocs_gen_files.open("schemas/provider.schema.json", "w") as f:
    f.write(render_json_schema(RawProvider, title="terok-executor provider YAML"))

with mkdocs_gen_files.open("schemas/routes.schema.json", "w") as f:
    schema = _routes_adapter.json_schema()
    schema.setdefault("title", "terok-executor generated routes.json")
    f.write(json.dumps(schema, indent=2))

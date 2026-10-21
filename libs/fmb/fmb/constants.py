# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""FMB package paths."""

import os

# libs/fmb/assets: the FMB board and peg USDs and the measured hole positions (clearance.json).
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

#!/usr/bin/env python3

import pandas as pd
from pony import orm

from openset.experiments.distributions import Generated

Generated.setup_db()

with orm.db_session():
    rows = Generated.Cache.select()
    df = pd.DataFrame(row.to_dict() for row in rows)

print(len(df))

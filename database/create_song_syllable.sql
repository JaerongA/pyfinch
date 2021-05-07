-- Store results from raster_syllable
CREATE TABLE IF NOT EXISTS song_syllable(
            syllableID      INTEGER PRIMARY KEY,
            clusterID       INTEGER NOT NULL,
            taskSession              INT,
            taskSessionDeafening     INT,
            taskSessionPostDeafening INT,
            dph                      INT,
            block10days              INT,
            note                     STRING,
            nbNote                   INT,
            nbSpk                    INTEGER,
            preMotorFR               REAL,
            psdSimilarity            REAL,
            pccUndir                 REAL,
            pccDir                   REAL,
            FOREIGN KEY(clusterID) REFERENCES cluster(id)
                                   )


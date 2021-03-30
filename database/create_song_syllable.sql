-- Store results from raster_syllable
CREATE TABLE IF NOT EXISTS song_syllable(
            syllableID      INTEGER PRIMARY KEY,
            clusterID       INTEGER NOT NULL,
            note            STRING,
            nbPremotorSpk INTEGER,
            psdSimilarity  REAL,
            FOREIGN KEY(clusterID) REFERENCES cluster(id)
                                   )
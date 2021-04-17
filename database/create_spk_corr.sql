
CREATE TABLE IF NOT EXISTS spk_corr(
            syllableID      INTEGER PRIMARY KEY,
            clusterID       INTEGER NOT NULL,
            note            STRING,
            nbPremotorFR    REAL,
            corrR           REAL,
            corrPval        REAL,
            corrSig         BOOLEAN,
            rsquare         REAL,
            FOREIGN KEY(clusterID) REFERENCES cluster(id)
                                   )
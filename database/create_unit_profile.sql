-- Create unit_profile table
-- Take values from cluster db

CREATE TABLE IF NOT EXISTS unit_profile(
            clusterID     INTEGER NOT NULL UNIQUE,
            birdID         STRING,
            taskName       STRING,
            taskSession    INTEGER,
            site           STRING,
            channel        STRING,
            unit           STRING,
            region         STRING,

            nbSpk                    INT,
            baselineFR               REAL,
            motifFRUndir             REAL,
            motifFRDir               REAL,
            spkHeight                REAL,
            spkWidth                 REAL,
            SNR                      REAL,

            unitCategoryUndir        STRING,
            unitCategoryDir          STRING,

            burstDurationBaseline    REAL,
            burstFreqBaseline        REAL,
            burstMeanNbSpkBaseline   REAL,
            burstFractionBaseline    REAL,
            burstIndexBaseline       REAL,

            burstDurationUndir       REAL,
            burstFreqUndir           REAL,
            burstMeanNbSpkUndir      REAL,
            burstFractionUndir       REAL,
            burstIndexUndir          REAL,

            burstDurationDir         REAL,
            burstFreqDir             REAL,
            burstMeanNbSpkDir        REAL,
            burstFractionDir         REAL,
            burstIndexDir            REAL,
            FOREIGN KEY(clusterID) REFERENCES cluster(id));

INSERT OR IGNORE INTO unit_profile (clusterID, birdID, taskName, taskSession, site, channel, unit, region)
SELECT id, birdID, taskName, taskSession, site, channel, unit, region FROM main.cluster
WHERE ephysOK is TRUE


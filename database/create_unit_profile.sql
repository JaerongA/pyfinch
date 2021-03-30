-- Store results from firing rates, waveform metrics, bursting, ISI analysis
CREATE TABLE IF NOT EXISTS unit_profile(
            clusterID     INTEGER NOT NULL UNIQUE,
            birdID         STRING,
            taskName       STRING,
            taskSession    INTEGER,
            site           STRING,
            channel        STRING,
            unit           STRING,
            region         STRING,
            SNR                      REAL,
            spkHeight                REAL,
            spkWidth                 REAL,
            spkHalfWidth             REAL,
            nbSpk                    INT,
            baselineFR               REAL,
            motifFRUndir             REAL,
            motifFRDir               REAL,
            unitCategoryBaseline     STRING,
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

            withinRefPropBaseline    REAL,
            isiPeakLatencyBaseline   REAL,
            isiCVBaseline            REAL,

            withinRefPropUndir       REAL,
            isiPeakLatencyUndir      REAL,
            isiCVUndir               REAL,

            withinRefPropDir         REAL,
            isiPeakLatencyDir        REAL,
            isiCVDir                 REAL,

            FOREIGN KEY(clusterID) REFERENCES cluster(id))
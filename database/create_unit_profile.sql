-- Store results from firing rates, waveform metrics, bursting, ISI analysis


CREATE TABLE unit_profile(
            cluster_id     INTEGER,
            birdID         STRING  not null,
            taskName       STRING  not null,
            taskSession    INTEGER not null,
            site           STRING  not null,
            channel        STRING  not null,
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

            withinRefProp            REAL,
            isiPeakLatencyBaseline   REAL,
            isiPeakLatencyUndir      REAL,
            isiPeakLatencyDir        REAL,
            FOREIGN KEY(cluster_id) REFERENCES cluster(id))




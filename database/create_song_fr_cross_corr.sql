-- Save results from song_fr_cross_corr.py


CREATE TABLE IF NOT EXISTS song_fr_cross_corr(
            clusterID     INTEGER NOT NULL UNIQUE,
            nbMotifUndir             INTEGER,  -- use only undir motifs for this analysis
            crossCorrMax             REAL,  -- maximum values of the mean cross-correlation
            peakLatency              REAL,  -- peak location of the mean cross-correlation
            FOREIGN KEY(clusterID) REFERENCES cluster(id));

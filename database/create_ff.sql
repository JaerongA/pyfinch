-- DB for FF (fundamental frequency analysis)

CREATE TABLE IF NOT EXISTS ff(
            id                       INTEGER PRIMARY KEY,
            birdID                   STRING,
            ffNote                   STRING,
            ffParameter              INT,  -- percent or ms from the start/end
            ffCriterion              STRING,   -- determine where to set FF start ('percent_from_start', 'ms_from_start', 'ms_from_end')
            ffLOW                    INT,   -- Lower limit of FF
            ffHigh                   INT,   -- Max limit of FF
            ffDuration               INT
            );

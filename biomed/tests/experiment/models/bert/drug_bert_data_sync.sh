#!/bin/sh
aws s3 sync s3://biomed-data/mimic/20190207_andy/txt/Consult /opt/S3/biomed-data/mimic/20190207_andy/txt/Consult
aws s3 sync s3://biomed-data/mimic/20190207_andy/txt/DischargeSummary/12 /opt/S3/biomed-data/mimic/20190207_andy/txt/DischargeSummary/12
aws s3 sync s3://biomed-data/mtsamples/clean /opt/S3/biomed-data/mtsamples/clean
aws s3 sync s3://biomed-data/briana.galloway/BIOMED-1215-summary /opt/S3/biomed-data/briana.galloway/BIOMED-1215-summary
aws s3 sync s3://biomed-data/deleys.brandman/annotation_BIOMED-655 /opt/S3/biomed-data/deleys.brandman/annotation_BIOMED-655
aws s3 sync s3://biomed-data/deleys.brandman/BIOMED-1991-subset /opt/S3/biomed-data/deleys.brandman/BIOMED-1991-subset

# sync the feature set
aws s3 sync s3://biomed-data/dev/20201012/a/mimic/20190207_andy/txt/Consult /opt/S3/biomed-data/dev/20201012/a/mimic/20190207_andy/txt/Consult
aws s3 sync s3://biomed-data/dev/20201012/a/mimic/20190207_andy/txt/DischargeSummary/12 /opt/S3/biomed-data/dev/20201012/a/mimic/20190207_andy/txt/DischargeSummary/12
aws s3 sync s3://biomed-data/dev/20201012/a/mtsamples/clean /opt/S3/biomed-data/dev/20201012/a/mtsamples/clean
aws s3 sync s3://biomed-data/dev/20201012/b/mimic/20190207_andy/txt/Consult /opt/S3/biomed-data/dev/20201012/b/mimic/20190207_andy/txt/Consult
aws s3 sync s3://biomed-data/dev/20201012/b/mimic/20190207_andy/txt/DischargeSummary/12 /opt/S3/biomed-data/dev/20201012/b/mimic/20190207_andy/txt/DischargeSummary/12
aws s3 sync s3://biomed-data/dev/20201012/b/mtsamples/clean /opt/S3/biomed-data/dev/20201012/b/mtsamples/clean
aws s3 sync s3://biomed-data/dev/20201012/c/mimic/20190207_andy/txt/Consult /opt/S3/biomed-data/dev/20201012/c/mimic/20190207_andy/txt/Consult
aws s3 sync s3://biomed-data/dev/20201012/c/mimic/20190207_andy/txt/DischargeSummary/12 /opt/S3/biomed-data/dev/20201012/c/mimic/20190207_andy/txt/DischargeSummary/12
aws s3 sync s3://biomed-data/dev/20201012/c/mtsamples/clean /opt/S3/biomed-data/dev/20201012/c/mtsamples/clean
aws s3 sync s3://biomed-data/dev/20201012/d/mimic/20190207_andy/txt/Consult /opt/S3/biomed-data/dev/20201012/d/mimic/20190207_andy/txt/Consult
aws s3 sync s3://biomed-data/dev/20201012/d/mimic/20190207_andy/txt/DischargeSummary/12 /opt/S3/biomed-data/dev/20201012/d/mimic/20190207_andy/txt/DischargeSummary/12
aws s3 sync s3://biomed-data/dev/20201012/d/mtsamples/clean /opt/S3/biomed-data/dev/20201012/d/mtsamples/clean
aws s3 sync s3://biomed-data/dev/20201012/e/mimic/20190207_andy/txt/Consult /opt/S3/biomed-data/dev/20201012/e/mimic/20190207_andy/txt/Consult
aws s3 sync s3://biomed-data/dev/20201012/e/mimic/20190207_andy/txt/DischargeSummary/12 /opt/S3/biomed-data/dev/20201012/e/mimic/20190207_andy/txt/DischargeSummary/12
aws s3 sync s3://biomed-data/dev/20201012/e/mtsamples/clean /opt/S3/biomed-data/dev/20201012/e/mtsamples/clean

package ai.jgp.drsti.spark.demo.airtraffic.lab400_monthly_air_traffic_to_delta;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.lit;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * @author jgp
 *
 */
public class MonthlyAirTrafficSaveDeltaApp {
  private static Logger log =
      LoggerFactory.getLogger(MonthlyAirTrafficSaveDeltaApp.class);

  public static void main(String[] args) {
    MonthlyAirTrafficSaveDeltaApp app = new MonthlyAirTrafficSaveDeltaApp();
    app.start();
  }

  /**
   * Real work goes here...
   */
  private boolean start() {
    log.debug("-> start()");
    long t0 = System.currentTimeMillis();

    // Creates a session on a local master
    SparkSession spark = SparkSession.builder()
        .appName("CSV to Dataset")
        .master("local[*]")
        .getOrCreate();

    long tc = System.currentTimeMillis();
    log.info("Spark master available in {} ms.", (tc - t0));
    t0 = tc;

    // Creates the schema
    StructType schema = DataTypes.createStructType(new StructField[] {
        DataTypes.createStructField(
            "month",
            DataTypes.DateType,
            false),
        DataTypes.createStructField(
            "pax",
            DataTypes.IntegerType,
            true) });

    // Reads a CSV file with header
    Dataset<Row> internationalPaxDf = spark.read().format("csv")
        .option("header", true)
        .option("dateFormat", "MMMM yyyy")
        .schema(schema)
        .load(
            "data/bts/International USCarrier_Traffic_20210902163435.csv");
    internationalPaxDf = internationalPaxDf
        .withColumnRenamed("pax", "internationalPax")
        // Very simple data quality
        .filter(col("month").isNotNull())
        .filter(col("internationalPax").isNotNull());

    tc = System.currentTimeMillis();
    log.info("International pax ingested in {} ms.", (tc - t0));
    t0 = tc;

    // Domestic
    Dataset<Row> domesticPaxDf = spark.read().format("csv")
        .option("header", true)
        .option("dateFormat", "MMMM yyyy")
        .schema(schema)
        .load(
            "data/bts/Domestic USCarrier_Traffic_20210902163435.csv");
    domesticPaxDf = domesticPaxDf
        .withColumnRenamed("pax", "domesticPax")
        // Very simple data quality
        .filter(col("month").isNotNull())
        .filter(col("domesticPax").isNotNull());
    tc = System.currentTimeMillis();
    log.info("Domestic pax ingested in {} ms.", (tc - t0));
    t0 = tc;

    // Combining datasets
    Dataset<Row> df = internationalPaxDf
        .join(domesticPaxDf,
            internationalPaxDf.col("month")
                .equalTo(domesticPaxDf.col("month")),
            "outer")
        .withColumn("pax", expr("internationalPax + domesticPax"))
        .drop(domesticPaxDf.col("month"))
        // Very simple data quality
        .filter(
            col("month").$less(lit("2020-01-01").cast(DataTypes.DateType)))
        .orderBy(col("month"))
        .cache();

    tc = System.currentTimeMillis();
    log.info("Transformation to gold zone in {} ms.", (tc - t0));
    t0 = tc;

    df.write()
        .format("delta")
        .mode("overwrite")
        .save("./data/tmp/airtrafficmonth");

    tc = System.currentTimeMillis();
    log.info("Data save for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }
}

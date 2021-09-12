package ai.jgp.drsti.spark.demo.airtraffic.lab410_monthly_air_traffic_delta;

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

import ai.jgp.drsti.spark.DrstiChart;
import ai.jgp.drsti.spark.DrstiK;
import ai.jgp.drsti.spark.DrstiLineChart;
import ai.jgp.drsti.spark.DrstiUtils;
import ai.jgp.drsti.spark.utils.DataframeUtils;

/**
 * 
 * @author jgp
 *
 */
public class MonthlyAirTrafficFromDeltaApp {
  private static Logger log =
      LoggerFactory.getLogger(MonthlyAirTrafficFromDeltaApp.class);

  public static void main(String[] args) {
    MonthlyAirTrafficFromDeltaApp app = new MonthlyAirTrafficFromDeltaApp();
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
        .appName("Read from Delta send to Drsti")
        .master("local[*]")
        .getOrCreate();

    long tc = System.currentTimeMillis();
    log.info("Spark master available in {} ms.", (tc - t0));
    t0 = tc;

    // Combining datasets
    Dataset<Row> df = spark.read().format("delta")
        .load("./data/tmp/airtrafficmonth")        .orderBy(col("month"))
;

    df = DrstiUtils.setHeader(df, "month", "Month of");
    df = DrstiUtils.setHeader(df, "pax", "Passengers");
    df = DrstiUtils.setHeader(
        df, "internationalPax", "International Passengers");
    df = DrstiUtils.setHeader(
        df, "domesticPax", "Domestic Passengers");

    tc = System.currentTimeMillis();
    log.info("Transformation to gold zone in {} ms.", (tc - t0));
    t0 = tc;

    // Shows at most 5 rows from the dataframe
    df.show(5, false);
    df.printSchema();

    DrstiChart d = new DrstiLineChart(df);
    d.setTitle("Air passenger traffic per month");
    d.setXScale(DrstiK.SCALE_TIME);
    d.setXTitle(
        "Period from " + DataframeUtils.min(df, "month") + " to "
            + DataframeUtils.max(df, "month") + " Data cached in Delta Lake.");
    d.setYTitle("Passengers (000s)");
    d.render();

    tc = System.currentTimeMillis();
    log.info("Data exported for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }
}

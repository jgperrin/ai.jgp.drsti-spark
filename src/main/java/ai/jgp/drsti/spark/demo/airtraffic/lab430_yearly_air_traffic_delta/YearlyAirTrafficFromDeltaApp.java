package ai.jgp.drsti.spark.demo.airtraffic.lab430_yearly_air_traffic_delta;

import static org.apache.spark.sql.functions.ceil;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.concat;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.month;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.year;

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
public class YearlyAirTrafficFromDeltaApp {
  private static Logger log =
      LoggerFactory.getLogger(YearlyAirTrafficFromDeltaApp.class);

  public static void main(String[] args) {
    YearlyAirTrafficFromDeltaApp app = new YearlyAirTrafficFromDeltaApp();
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

    // Reading gold data from Delta
    Dataset<Row> df = spark.read().format("delta")
        .load("./data/tmp/airtrafficmonth")
        .orderBy(col("month"));

    tc = System.currentTimeMillis();
    log.info("Reading gold zone in {} ms.", (tc - t0));
    t0 = tc;

    Dataset<Row> dfYear = df
        .withColumn("year", year(col("month")))
        .groupBy(col("year"))
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"))
        .orderBy(col("year"));
    dfYear = DrstiUtils.setHeader(dfYear, "year", "Year");
    dfYear = DrstiUtils.setHeader(dfYear, "pax", "Passengers");
    dfYear = DrstiUtils.setHeader(
        dfYear, "internationalPax", "International Passengers");
    dfYear = DrstiUtils.setHeader(
        dfYear, "domesticPax", "Domestic Passengers");
    tc = System.currentTimeMillis();
    log.info("Transformation for yearly graph {} ms.", (tc - t0));
    t0 = tc;

    dfYear.show(5, false);
    dfYear.printSchema();

    DrstiLineChart d = new DrstiLineChart(dfYear);
    d.setTitle("US air traffic, in passengers, per year");
    d.setXTitle("Year " + DataframeUtils.min(dfYear, "year") + " - " +
        DataframeUtils.max(dfYear, "year")+ " - Data cached in Delta Lake");
    d.setYTitle("Passengers (000s)");
    d.render();

    tc = System.currentTimeMillis();
    log.info("Data exported for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }
}

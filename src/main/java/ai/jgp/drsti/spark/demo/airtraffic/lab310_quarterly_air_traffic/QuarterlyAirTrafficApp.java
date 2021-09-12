package ai.jgp.drsti.spark.demo.airtraffic.lab310_quarterly_air_traffic;

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
public class QuarterlyAirTrafficApp {
  private static Logger log =
      LoggerFactory.getLogger(QuarterlyAirTrafficApp.class);

  public static void main(String[] args) {
    QuarterlyAirTrafficApp app = new QuarterlyAirTrafficApp();
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

    Dataset<Row> dfQuarter = df
        .withColumn("year", year(col("month")))
        .withColumn("q", ceil(month(col("month")).$div(3)))
        .withColumn("period", concat(col("year"), lit("-Q"), col("q")))
        .groupBy(col("period"))
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"))
        .drop("year")
        .drop("q")
        .orderBy(col("period"));

    dfQuarter = DrstiUtils.setHeader(dfQuarter, "period", "Quarter");
    dfQuarter = DrstiUtils.setHeader(dfQuarter, "pax", "Passengers");
    dfQuarter = DrstiUtils.setHeader(
        dfQuarter, "internationalPax", "International Passengers");
    dfQuarter = DrstiUtils.setHeader(
        dfQuarter, "domesticPax", "Domestic Passengers");

    dfQuarter.show(5, false);
    dfQuarter.printSchema();

    // Shows at most 5 rows from the dataframe
    df.show(5, false);
    df.printSchema();

    DrstiChart d = new DrstiLineChart(dfQuarter);
    d.setTitle("Air passenger traffic per quarter");
    d.setXScale(DrstiK.SCALE_LABELS);
    d.setXTitle(
        "Period from " + DataframeUtils.min(df, "month") + " to "
            + DataframeUtils.max(df, "month"));
    d.setYTitle("Passengers (000s)");
    d.render();

    tc = System.currentTimeMillis();
    log.info("Data exported for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }
}

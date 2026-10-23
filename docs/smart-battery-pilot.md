# Using it with Smart Battery Pilot

[Smart Battery Pilot](https://cygnusb.github.io/ha-smart-battery-pilot/) plans
a home battery from dynamic prices. It reads Spot Forecast without any change
on either side, and neither project imports the other.

## Setup

In Smart Battery Pilot's price step (or later in its options), choose

```text
sensor.spot_forecast_de_lu_price
```

as the price entity. Its ENTSO-E adapter detects the `prices` attribute with
`time` and `price` keys and reads it as it is. The entity id is the same on
every instance language.

## What Smart Battery Pilot gets

* The known prices of today and tomorrow, at their native resolution
  (15 minutes), followed by hourly forecast slots up to six days after the
  last known day. Each slot ends where the next one starts, which is how the
  adapter reads it.
* Forecast slots carry `source: forecast`, the horizon `h`, and the peak-risk
  (`p90`) and low-price (`p10`) values. The adapter ignores these keys; they
  are there for templates and for future use.
* Prices are the bare exchange price in EUR/kWh. Smart Battery Pilot adds its
  own `price_offset` for grid fees and taxes, as with any other price source.

The adapter does not tell forecast slots from known ones. In the morning,
before tomorrow's prices are published, Smart Battery Pilot therefore sees
prices for tomorrow too, from the forecast. Whether it should plan on forecast
slots differently, for example hold charge before a likely peak using `p90`,
is a question for Smart Battery Pilot; Spot Forecast only guarantees the data
is there.

## Mind the peaks

The expected price underestimates the height of price peaks (see
[accuracy](accuracy.md#peaks)). A plan built on forecast hours may hold
somewhat less charge for a peak than the actual price would justify. Once the
auction result is published, the known prices replace the forecast and the
plan follows them.

"""
Support rep front end for the booking cancellation retention workflow.

The rep fills in a booking profile and clicks Analyze. This sends the
profile to the n8n webhook, which calls the deployed FastAPI model for a
cancellation probability, passes the result to an AI agent that decides
on a risk tier, a retention offer, and a drafted message, and returns
all three back here in the same request.

Field placement in this form follows the feature importance and Cramer's V
results from the notebook: fields shown in the main form
are the ones that most influence the model's prediction and are also
realistically known by a rep at the time of the call. Everything else is
in the Advanced section. All fields the model accepts are collected and
sent on every request; none are left for the API to silently default.

Run with:
    streamlit run streamlit_app.py
"""

from datetime import date, timedelta

import requests
import streamlit as st

N8N_WEBHOOK_URL = "https://dsc2302025.app.n8n.cloud/webhook/cancellation-analysis"

st.set_page_config(page_title="Booking Cancellation Risk Assistant", layout="centered")
st.title("Booking Cancellation Risk Assistant")
st.caption("Enter a booking profile to get a cancellation risk assessment and a suggested retention response.")

with st.form("booking_profile_form"):
    st.subheader("Booking Profile")
    st.caption("These fields have the strongest influence on the prediction, including the guest's "
               "country, booking agent, and arrival month — all required by the model.")

    col1, col2 = st.columns(2)
    with col1:
        hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input(
            "Lead time (days before arrival)", min_value=0, value=60,
            help="The single strongest predictor of cancellation in the model."
        )
        arrival_date = st.date_input(
            "Arrival date", value=date.today() + timedelta(days=60),
            help="Drives the model's arrival_date_year/month/week/day features. "
                 "Cancellation risk rises in the summer peak season (Section 5.4)."
        )
        adr = st.number_input("Average daily rate", min_value=0.0, value=120.0, step=5.0)
        stays_in_weekend_nights = st.number_input("Weekend nights (Fri/Sat)", min_value=0, value=1)
        stays_in_week_nights = st.number_input("Week nights (Sun-Thu)", min_value=0, value=2)
        adults = st.number_input("Adults", min_value=1, value=2)
        children = st.number_input("Children", min_value=0, value=0)
        market_segment = st.selectbox(
            "Market segment",
            ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation"],
            help="The strongest categorical predictor tested (Cramer's V 0.221)."
        )
    with col2:
        customer_type = st.selectbox(
            "Customer type", ["Transient", "Contract", "Transient-Party", "Group"]
        )
        deposit_type = st.selectbox(
            "Deposit type", ["No Deposit", "Non Refund", "Refundable"],
            help="Non refundable deposits show a much higher cancellation rate in this data."
        )
        country = st.text_input(
            "Country code", value="PRT",
            help="Ranked 2nd in the model's feature importance ranking (Section 6.7), "
                 "ahead of deposit_type. Always enter the guest's real country code; leaving "
                 "this at a placeholder would override one of the model's strongest signals."
        )
        agent = st.number_input(
            "Booking agent ID (0 if none)", min_value=0, value=0,
            help="Ranked 6th in the model's feature importance ranking (Section 6.7). "
                 "Always enter the real booking agent ID."
        )
        previous_cancellations = st.number_input(
            "Previous cancellations by this guest", min_value=0, value=0,
            help="Guests with at least one prior cancellation cancel at roughly 2.5x the rate of guests with none."
        )
        is_repeated_guest = st.selectbox(
            "Repeat guest", [0, 1], index=0,
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Returning guests cancel far less often than first-time guests."
        )
        booking_changes = st.number_input("Booking changes made so far", min_value=0, value=0)
        total_of_special_requests = st.number_input(
            "Special requests", min_value=0, value=0,
            help="More special requests is associated with a lower cancellation rate."
        )
        required_car_parking_spaces = st.number_input(
            "Car parking spaces required", min_value=0, value=0,
            help="In this data, every booking that requested parking was honoured."
        )

    with st.expander("Advanced fields (lower impact on the prediction, sensible defaults are used if left as is)"):
        st.caption(
            "These fields were shown in the notebook's EDA to matter less than the "
            "ones above. Leaving them at their defaults will not meaningfully "
            "change the result."
        )
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            meal = st.selectbox(
                "Meal plan", ["BB", "HB", "FB", "SC"], index=0,
                help="Lowest Cramer's V of all categorical features tested (0.064)."
            )
            distribution_channel = st.selectbox(
                "Distribution channel", ["Direct", "Corporate", "TA/TO", "GDS"], index=0,
                help="Overlaps conceptually with market segment, which is already captured above."
            )
            previous_bookings_not_canceled = st.number_input(
                "Previous bookings not cancelled", min_value=0, value=0
            )
        with adv_col2:
            days_in_waiting_list = st.number_input("Days in waiting list", min_value=0, value=0)
            babies = st.number_input(
                "Babies", min_value=0, value=0,
                help="Rare and low-signal in the notebook's EDA."
            )

    submitted = st.form_submit_button("Analyze")

if submitted:
    total_nights = int(stays_in_weekend_nights) + int(stays_in_week_nights)

    payload = {
        "hotel": hotel,
        "lead_time": int(lead_time),
        "adr": float(adr),
        "total_nights": total_nights,
        "stays_in_weekend_nights": int(stays_in_weekend_nights),
        "stays_in_week_nights": int(stays_in_week_nights),
        "adults": int(adults),
        "children": int(children),
        "babies": int(babies),
        "market_segment": market_segment,
        "customer_type": customer_type,
        "deposit_type": deposit_type,
        "previous_cancellations": int(previous_cancellations),
        "is_repeated_guest": int(is_repeated_guest),
        "booking_changes": int(booking_changes),
        "total_of_special_requests": int(total_of_special_requests),
        "required_car_parking_spaces": int(required_car_parking_spaces),
        "arrival_date_year": arrival_date.year,
        "arrival_date_month": arrival_date.strftime("%B"),
        "arrival_date_week_number": arrival_date.isocalendar()[1],
        "arrival_date_day_of_month": arrival_date.day,
        "meal": meal,
        "country": country,
        "distribution_channel": distribution_channel,
        "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
        "agent": int(agent),
        "days_in_waiting_list": int(days_in_waiting_list),
    }

    with st.spinner("Analyzing booking profile..."):
        try:
            response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the analysis workflow: {e}")
            result = None

    if result:
        st.subheader("Result")

        risk_tier = result.get("risk_tier", "Unknown")
        cancellation_probability = result.get("cancellation_probability")
        retention_offer = result.get("retention_offer", "No offer suggested")
        drafted_message = result.get("drafted_message", "")

        tier_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk_tier, "grey")
        st.markdown(f"**Risk tier:** :{tier_color}[{risk_tier}]")
        if cancellation_probability is not None:
            st.markdown(f"**Cancellation probability:** {cancellation_probability:.1%}")

        st.markdown("**Suggested retention offer**")
        st.info(retention_offer)

        st.markdown("**Drafted message to customer**")
        st.text_area("Message", value=drafted_message, height=150, label_visibility="collapsed")

        if risk_tier == "High":
            st.warning("This case has been automatically logged for manager review.")

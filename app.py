import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse

MODEL_FILES = ["model.pkl", "best_model.pkl"]
model_path = next((Path(name) for name in MODEL_FILES if Path(name).exists()), None)

st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🔒",
    layout="wide",
)

st.sidebar.title("🔒 Phishing URL Detection")
st.sidebar.write(
    "Enter a URL to identify if it is a phishing or legitimate website."
)
st.sidebar.markdown(
    "**How to use:**\n"
    "1. Enter the URL.\n"
    "2. Click **Predict URL**.\n"
    "3. Review the extracted features and classification results."
)
st.sidebar.info("Target label: 1 = Legitimate, 0 = Phishing")
st.sidebar.caption("Model files supported: model.pkl or best_model.pkl")

stickfile_path = Path("stickfile.txt")
if stickfile_path.exists():
    st.sidebar.download_button(
        label="Download stickfile",
        data=stickfile_path.read_text(encoding="utf-8"),
        file_name="stickfile.txt",
        mime="text/plain",
    )

st.title("Phishing URL Detection")
st.write(
    "This app classifies a single URL as Legitimate or Phishing. "
    "Enter the URL and click Predict."
)

if model_path is None:
    st.error(
        "No trained model file was found. Please add `model.pkl` or `best_model.pkl` to the project folder."
    )
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as exc:
    st.error(f"Failed to load model from `{model_path.name}`: {exc}")
    st.stop()

COMMON_TLDS = {
    "com", "net", "org", "edu", "gov", "info", "io", "co", "us", "uk",
    "de", "fr", "ru", "jp", "in", "cn", "biz", "online", "site", "app",
    "tech", "store", "xyz", "top", "loan", "win", "live",
}

FEATURE_COLUMNS = [
    "URL", "URLLength", "Domain", "DomainLength", "IsDomainIP", "TLD",
    "URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
    "TLDLength", "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar",
    "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL",
    "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS", "LineOfCode",
    "LargestLineLength", "HasTitle", "Title", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive", "NoOfURLRedirect",
    "NoOfSelfRedirect", "HasDescription", "NoOfPopup", "NoOfiFrame",
    "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
    "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo",
    "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef",
    "NoOfExternalRef",
]

URL_REGEX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def normalize_url(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if not URL_REGEX.match(value):
        value = "http://" + value
    return value


def is_ip_address(domain: str) -> int:
    return int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", domain)))


def longest_char_run(value: str) -> int:
    if not value:
        return 0
    max_run = 1
    current = 1
    for a, b in zip(value, value[1:]):
        if a == b:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def similarity_ratio(a: str, b: str) -> float:
    a = str(a or "").lower()
    b = str(b or "").lower()
    if not a or not b:
        return 0.0
    tokens = re.findall(r"[a-z0-9]+", a)
    if not tokens:
        return 0.0
    common = sum(1 for token in tokens if token in b)
    return min(100.0, 100.0 * common / len(tokens))


def count_special_chars(value: str) -> int:
    return sum(1 for ch in value if not ch.isalnum() and ch not in "./:?&=#%-_@+")


def extract_domain_and_tld(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().split(":")[0].lstrip("www.")
    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    return parsed.scheme.lower(), domain, tld


def fetch_page(url: str) -> tuple[str, requests.Response]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
    response.raise_for_status()
    return response.text, response


def parse_page_features(html: str, url: str, domain: str, response: requests.Response) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_text = soup.title.string.strip() if soup.title and soup.title.string else ""
    has_title = int(bool(title_text))
    description_meta = soup.find("meta", attrs={"name": "description"})
    has_description = int(bool(description_meta and description_meta.get("content", "").strip()))
    has_favicon = int(bool(soup.find("link", rel=lambda x: x and "icon" in x.lower())))
    has_robots = int(bool(soup.find("meta", attrs={"name": "robots"})))
    is_responsive = int(bool(soup.find("meta", attrs={"name": "viewport"})))
    page_text = soup.get_text(separator=" ").strip()
    bank = int(bool(re.search(r"\bbank\b", page_text, re.I)))
    pay = int(bool(re.search(r"\bpay\b|\bpayment\b|\bpaypal\b", page_text, re.I)))
    crypto = int(bool(re.search(r"\bcrypto\b|\bbitcoin\b|\bethereum\b", page_text, re.I)))
    has_copyright = int(bool(re.search(r"©|copyright", page_text, re.I)))
    no_images = len(soup.find_all("img"))
    no_css = len([tag for tag in soup.find_all("link", rel=True) if "stylesheet" in " ".join(tag["rel"]).lower()])
    no_js = len([tag for tag in soup.find_all("script") if tag.get("src")])
    iframes = soup.find_all("iframe")
    no_iframes = len(iframes)
    no_popup = len(re.findall(r"window\.open\(|alert\(|confirm\(|prompt\(", html, re.I))

    forms = soup.find_all("form")
    has_external_form_submit = 0
    has_submit_button = 0
    has_hidden_fields = 0
    has_password_field = 0
    for form in forms:
        action = (form.get("action", "") or "").strip()
        if action and action.startswith("http") and domain not in action:
            has_external_form_submit = 1
        if form.find("input", attrs={"type": "submit"}) or form.find("button", attrs={"type": "submit"}):
            has_submit_button = 1
        if form.find("input", attrs={"type": "hidden"}):
            has_hidden_fields = 1
        if form.find("input", attrs={"type": "password"}):
            has_password_field = 1

    anchors = soup.find_all("a", href=True)
    no_self_ref = 0
    no_empty_ref = 0
    no_external_ref = 0
    for a in anchors:
        href = a["href"].strip()
        if not href or href == "#":
            no_empty_ref += 1
        elif href.startswith("http"):
            if domain in href:
                no_self_ref += 1
            else:
                no_external_ref += 1
        else:
            no_self_ref += 1

    url_title_match_score = similarity_ratio(url, title_text)
    domain_title_match_score = similarity_ratio(domain, title_text)
    lines = html.splitlines()
    line_of_code = len(lines)
    largest_line_length = max((len(line) for line in lines), default=0)
    redirects = len(response.history)
    self_redirects = sum(1 for r in response.history if urlparse(r.url).netloc.lower().startswith(domain))

    return {
        "LineOfCode": line_of_code,
        "LargestLineLength": largest_line_length,
        "HasTitle": has_title,
        "Title": title_text,
        "DomainTitleMatchScore": domain_title_match_score,
        "URLTitleMatchScore": url_title_match_score,
        "HasFavicon": has_favicon,
        "Robots": has_robots,
        "IsResponsive": is_responsive,
        "NoOfURLRedirect": redirects,
        "NoOfSelfRedirect": self_redirects,
        "HasDescription": has_description,
        "NoOfPopup": no_popup,
        "NoOfiFrame": no_iframes,
        "HasExternalFormSubmit": has_external_form_submit,
        "HasSocialNet": int(bool(re.search(r"facebook|twitter|instagram|linkedin|youtube|telegram|whatsapp", page_text, re.I))),
        "HasSubmitButton": has_submit_button,
        "HasHiddenFields": has_hidden_fields,
        "HasPasswordField": has_password_field,
        "Bank": bank,
        "Pay": pay,
        "Crypto": crypto,
        "HasCopyrightInfo": has_copyright,
        "NoOfImage": no_images,
        "NoOfCSS": no_css,
        "NoOfJS": no_js,
        "NoOfSelfRef": no_self_ref,
        "NoOfEmptyRef": no_empty_ref,
        "NoOfExternalRef": no_external_ref,
    }


def extract_url_features(url: str) -> pd.DataFrame:
    url = normalize_url(url)
    if not url:
        raise ValueError("URL cannot be empty.")

    scheme, domain, tld = extract_domain_and_tld(url)
    url_length = len(url)
    domain_length = len(domain)
    is_https = int(scheme == "https")
    is_domain_ip = is_ip_address(domain)
    url_similarity_index = int(similarity_ratio(domain, url))
    char_continuation_rate = longest_char_run(url) / max(1, url_length)
    tld_legit_prob = 100.0 if tld in COMMON_TLDS else 10.0
    url_char_prob = sum(1 for ch in url if ch.isalnum()) / max(1, url_length)
    no_of_subdomain = max(0, domain.count(".") - 1)
    special_chars = ["@", "%", "_", "-", "+", "#", "!", "$", "^", "*", ";", ":", "\\"]
    no_of_obfuscated_char = sum(url.count(ch) for ch in special_chars)
    has_obfuscation = int("@" in url or "%" in url or "+" in url)
    obfuscation_ratio = no_of_obfuscated_char / max(1, url_length)
    no_of_letters = sum(1 for ch in url if ch.isalpha())
    no_of_digits = sum(1 for ch in url if ch.isdigit())
    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")
    no_of_other_special_chars = count_special_chars(url)
    special_char_ratio = no_of_other_special_chars / max(1, url_length)

    features = {
        "URL": url,
        "URLLength": url_length,
        "Domain": domain,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "TLD": tld,
        "URLSimilarityIndex": url_similarity_index,
        "CharContinuationRate": round(char_continuation_rate, 6),
        "TLDLegitimateProb": round(tld_legit_prob, 6),
        "URLCharProb": round(url_char_prob, 6),
        "TLDLength": len(tld),
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated_char,
        "ObfuscationRatio": round(obfuscation_ratio, 6),
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": round(no_of_letters / max(1, url_length), 6),
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": round(no_of_digits / max(1, url_length), 6),
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special_chars,
        "SpacialCharRatioInURL": round(special_char_ratio, 6),
        "IsHTTPS": is_https,
    }

    try:
        html, response = fetch_page(url)
        features.update(parse_page_features(html, url, domain, response))
    except Exception:
        features.update({
            "LineOfCode": 0,
            "LargestLineLength": 0,
            "HasTitle": 0,
            "Title": "",
            "DomainTitleMatchScore": 0.0,
            "URLTitleMatchScore": 0.0,
            "HasFavicon": 0,
            "Robots": 0,
            "IsResponsive": 0,
            "NoOfURLRedirect": 0,
            "NoOfSelfRedirect": 0,
            "HasDescription": 0,
            "NoOfPopup": 0,
            "NoOfiFrame": 0,
            "HasExternalFormSubmit": 0,
            "HasSocialNet": 0,
            "HasSubmitButton": 0,
            "HasHiddenFields": 0,
            "HasPasswordField": 0,
            "Bank": 0,
            "Pay": 0,
            "Crypto": 0,
            "HasCopyrightInfo": 0,
            "NoOfImage": 0,
            "NoOfCSS": 0,
            "NoOfJS": 0,
            "NoOfSelfRef": 0,
            "NoOfEmptyRef": 0,
            "NoOfExternalRef": 0,
        })

    return pd.DataFrame([features])


def prepare_prediction_input(data: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        feature_names = [name for name in model.feature_names_in_ if name in data.columns]
        if feature_names:
            return data[feature_names]
    fallback = data.drop(columns=[col for col in ["URL", "Domain", "TLD", "Title"] if col in data.columns], errors="ignore")
    if not fallback.empty:
        return fallback
    return data.select_dtypes(include=[np.number])


def predict_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    X = prepare_prediction_input(result)
    try:
        predictions = model.predict(X)
    except Exception:
        numeric = result.select_dtypes(include=[np.number])
        predictions = model.predict(numeric)
    result["Prediction"] = ["Legitimate" if int(p) == 1 else "Phishing" for p in predictions]
    return result


url_input = st.text_input("Enter a URL to classify", placeholder="https://example.com")
if st.button("Predict URL"):
    if not url_input.strip():
        st.warning("Please enter a valid URL before predicting.")
    else:
        with st.spinner("Extracting features and predicting..."):
            try:
                feature_df = extract_url_features(url_input)
                prediction_df = predict_dataframe(feature_df)
                classification = prediction_df.loc[0, "Prediction"]

                st.subheader("Prediction Result")
                if classification == "Legitimate":
                    st.success("Result: Legitimate URL")
                else:
                    st.error("Result: Phishing URL")

                st.subheader("Extracted URL Features")
                st.dataframe(feature_df[FEATURE_COLUMNS].transpose(), use_container_width=True)

                st.subheader("Prediction Output")
                st.write(prediction_df.to_dict(orient="records")[0])
            except Exception as exc:
                st.error("Could not classify the URL. Please check the URL format and try again.")
                st.write(f"Details: {exc}")

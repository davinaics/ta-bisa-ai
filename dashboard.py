import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk, joblib, ast
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import altair as alt 

# Konfigurasi halaman 
st.set_page_config(page_title="Customer Review Dashboard", layout="wide")
st.title("Customer Review Dashboard 👥💬")

# Sidebar Navigation dengan style 
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["EDA", "Modelling", "Chatbot"],
        icons=["bar-chart", "cpu", "robot"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#000000"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "white",
            },
            "nav-link-selected": {
                "background-color": "maroon",
                "color": "white",
            },
            "icon-selected": {
                "color": "white"
            }
        }
    )

# Load files 
df_full = pd.read_csv("cleaned_dataset_full.csv")
df_model = pd.read_csv("cleaned_dataset_model.csv")

# Parse token list bila masih string
for df in [df_full, df_model]:
    if df_model["Customer Review Clean"].apply(lambda x: isinstance(x, str)).any():
        df_model["Customer Review Clean"] = df_model["Customer Review Clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

# Load TF-IDF dan model untuk tab Modelling
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("rf_model.pkl")

# ============================================================
# 1️⃣ EDA TAB
# ============================================================
if selected == "EDA":
    st.header("🔎 Exploratory Data Analysis")
    maroon = "#550000"
    darkgreen = "darkgreen"

    # --- Card Chart ---
    st.subheader("📊 Card Chart")
    def colored_card(title, value, color, icon):
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:15px;
                border-radius:15px;
                text-align:center;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                height:200px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
            ">
                <div style="
                    display:flex; 
                    align-items:center; 
                    justify-content:center; 
                    gap:6px;
                    flex-wrap:wrap;
                ">
                    <span style="font-size:24px; line-height:1;">{icon}</span>
                    <span style="color:white; font-size:20px; font-weight:600; line-height:1;">
                        {title}
                    </span>
                </div>
                <p style="
                    color:white;
                    font-size:24px;
                    font-weight:bold;
                    margin:10px 0 0 0;
                    white-space:nowrap;
                    overflow:hidden;
                    text-overflow:ellipsis;
                    text-align:center;
                ">
                    {value}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # KPI Cards 
    df_unique_products = df_full.drop_duplicates(
        subset=["Product Name", "Category", "Location", "Price", "Number Sold"]
    )

    total_sold = df_unique_products["Number Sold"].sum()
    total_products = df_unique_products["Product Name"].nunique()
    total_categories = df_unique_products["Category"].nunique()
    total_reviews = len(df_full)  
    avg_rating = f"{df_full['Customer Rating'].mean():.2f}"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        colored_card("Total Produk", total_products, "#006400", "📦")
    with col2:
        colored_card("Total Kategori", total_categories, "#550000", "🏷️")
    with col3:
        colored_card("Total Terjual", f"{total_sold:,}", "#006400", "🛒")
    with col4:
        colored_card("Total Review", f"{total_reviews:,}", "#550000", "📝")
    with col5:
        colored_card("AVG Rating", avg_rating, "#006400", "⭐")

    # Distribusi Sentimen 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Distribusi Sentimen")
        sentiment_counts = df_full["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Jumlah"]

        # hitung persen
        sentiment_counts["Persen"] = sentiment_counts["Jumlah"] / sentiment_counts["Jumlah"].sum()

        # bikin kolom label gabungan
        sentiment_counts["Label"] = sentiment_counts.apply(
            lambda row: f"{row['Jumlah']:,} ({row['Persen']:.1%})", axis=1
        )

        # warna maroon & hijau tua
        colors_map = {"Positive": "#006400", "Negative": "#550000"}
        color_scale = alt.Scale(
            domain=list(colors_map.keys()),
            range=list(colors_map.values())
        )

        # pie chart
        base = (
            alt.Chart(sentiment_counts)
            .encode(
                theta="Jumlah:Q",
                color=alt.Color("Sentiment:N", scale=color_scale, legend=None)
            )
        )

        chart_sent = (
            base.mark_arc(outerRadius=105, innerRadius=35)
            .encode(
                tooltip=[
                    alt.Tooltip("Sentiment:N", title="Sentiment"),
                    alt.Tooltip("Jumlah:Q", title="Jumlah", format=","),
                    alt.Tooltip("Persen:Q", title="Persen", format=".1%")
                ]
            )
        )

        # label jumlah + persen di tengah slice
        text = (
            alt.Chart(sentiment_counts)
            .mark_text(size=11, fontWeight="bold", color="white", radius=70)
            .encode(
                text="Label:N",
                theta=alt.Theta("Jumlah:Q", stack=True)
            )
        )

        final_chart = (chart_sent + text).properties(
            width=260, height=260, background="transparent"
        ).configure_view(stroke=None)

        st.altair_chart(final_chart, use_container_width=True)


    with col2:
        st.subheader("😊 Top Emosi Review")

        # Ambil top 10 emosi, lalu rapikan kolom
        emo = (
            df_full["Emotion"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        # Setelah reset_index, kolom otomatis jadi ["index", "Emotion"]
        emo.columns = ["Emotion", "Jumlah"]

        # skala warna hijau
        greens = sns.light_palette("#006400", n_colors=len(emo), reverse=True).as_hex()
        emo_domain = emo["Emotion"].tolist()

        chart_emo = (
            alt.Chart(emo)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Emotion:N",
                    title="Emosi",
                    sort="y",
                    axis=alt.Axis(labelAngle=30, labelColor="white", titleColor="white")
                ),
                y=alt.Y(
                    "Jumlah:Q",
                    title="Jumlah",
                    axis=alt.Axis(labelColor="white", titleColor="white")
                ),
                color=alt.Color("Emotion:N",
                            scale=alt.Scale(domain=emo_domain, range=greens),
                            legend=None),
                tooltip=[
                    alt.Tooltip("Emotion:N", title="Emosi"),
                    alt.Tooltip("Jumlah:Q", title="Jumlah", format=",")
                ]
            )
            .properties(height=300, width="container", background="transparent")
            .configure_view(stroke=None)
        )
        st.altair_chart(chart_emo, use_container_width=True)

    # Top 5 Kota & Kategori 
    with st.container():
        col1, col2 = st.columns([1, 1], gap="medium")

        # Top Kota
        with col1:
            st.subheader("🏙️ Top 5 Kota dengan Penjualan Terbanyak")

            sales_per_city = (
                df_unique_products.groupby('Location')['Number Sold']
                .sum()
                .reset_index()
                .sort_values(by='Number Sold', ascending=False)
            )
            top_city = sales_per_city.head(5)
            greens = sns.light_palette("#006400", n_colors=len(top_city), reverse=True).as_hex()
            color_scale = alt.Scale(domain=top_city["Location"], range=greens)

            chart_city = (
                alt.Chart(top_city)
                .mark_bar(color=darkgreen)
                .encode(
                    x=alt.X("Number Sold:Q", title="Jumlah Terjual"),
                    y=alt.Y("Location:N", sort="-x", title="Kota"),
                    tooltip=[
                        alt.Tooltip("Location:N", title="Kota"),
                        alt.Tooltip("Number Sold:Q", title="Jumlah Terjual", format=",")
                    ],
                    color=alt.Color("Location:N", scale=color_scale, legend=None),
                )
                .properties(width="container", height=300)
                .configure_view(stroke=None)
                .configure_axis(labelFontSize=12, titleFontSize=12)
            )

            st.altair_chart(chart_city, use_container_width=True)

        # Top Kategori
        with col2:
            st.subheader("📦 Top 5 Kategori dengan Penjualan Terbanyak")

            sales_per_cat = (
                df_unique_products.groupby('Category')['Number Sold']
                .sum()
                .reset_index()
                .sort_values(by='Number Sold', ascending=False)
            )
            top5_cat = sales_per_cat.head(5)

            # gradasi maroon -> light pink
            reds = sns.light_palette("maroon", n_colors=len(top5_cat), reverse=True).as_hex()
            color_scale = alt.Scale(domain=top5_cat["Category"], range=reds)

            chart_cat = (
                alt.Chart(top5_cat)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Category:N",
                        title="Kategori",
                        sort=alt.SortField("Number Sold", order="ascending"),
                        axis=alt.Axis(labelAngle=-30, labelFontSize=11, titleFontSize=14)
                    ),
                    y=alt.Y(
                        "Number Sold:Q",
                        title="Jumlah Terjual",
                        axis=alt.Axis(labelFontSize=11, titleFontSize=14)
                    ),
                    color=alt.Color("Category:N", scale=color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip("Category:N", title="Kategori"),
                        alt.Tooltip("Number Sold:Q", title="Jumlah Terjual", format=",")
                    ]
                )
                .properties(width="container", height=300, background="transparent")
                .configure_view(stroke=None)
            )

            st.altair_chart(chart_cat, use_container_width=True)


    # Top 5 Produk Terlaris 
    top5_products = (
        df_unique_products.groupby("Product Name")["Number Sold"]
        .sum()
        .reset_index()
        .sort_values(by="Number Sold", ascending=False)
        .head(5)
    )

    st.subheader("💰 Top 5 Produk dengan Penjualan Terbanyak", anchor="start")

    # gradasi maroon -> light pink
    reds = sns.light_palette("maroon", n_colors=len(top5_products), reverse=True).as_hex()
    color_scale = alt.Scale(domain=top5_products["Product Name"], range=reds)

    chart = (
        alt.Chart(top5_products)
        .mark_bar()
        .encode(
            x=alt.X(
                "Number Sold:Q",
                title="Jumlah Terjual",
                axis=alt.Axis(
                    labelColor="white",
                    titleColor="white",
                    labelFontSize=11,
                    titleFontSize=16
                )
            ),
            y=alt.Y(
                "Product Name:N",
                sort="-x",
                title="Produk",
                axis=alt.Axis(
                    labelColor="white",
                    titleColor="white",
                    labelFontSize=11,
                    titleFontSize=16,
                    labelLimit=300,
                    titleX=-350,
                )
            ),
            color=alt.Color("Product Name:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("Product Name:N", title="Produk"),
                alt.Tooltip("Number Sold:Q", title="Jumlah Terjual", format=",")
            ]
        )
        .properties(
            height=300,
            width="container",
            background="transparent"
        )
        .configure_view(stroke=None)
    )

    st.altair_chart(chart, use_container_width=True)

# ============================================================
# 2️⃣ MODELLING TAB
# ============================================================
elif selected == "Modelling":
    st.header("🤖 Modelling & Evaluation")
    st.caption("Model yang Digunakan: **Random Forest Classifier**")

    # Data Preparation 
    X_all = df_model["Customer Review Clean"].apply(lambda x: " ".join(x)) # digabung jd string
    y_all = df_model["Sentiment"]
    X_vec = tfidf.transform(X_all)

    # Split data 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Evaluasi di test set 
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Overall Accuracy", f"{acc:.2%}")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    st.subheader("📊 Classification Report")
    st.table(report_df)

    # Confusion Matrix 
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(2.8, 2.2), dpi=80, facecolor="#f3f5e7")
    custom_maroon = mcolors.LinearSegmentedColormap.from_list(
        "custom_maroon", ["#330000", "#550000", "#FFB6C1"]
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=custom_maroon,
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        annot_kws={"size": 7, "weight": "bold", "color": "white"},
        cbar_kws={'shrink': 0.4, 'aspect': 10},
        ax=ax
    )
    for text in ax.texts:
        text.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground="black"),
            path_effects.Normal()
        ])
    ax.set_xlabel("Predicted", fontsize=7, color="black", weight="bold")
    ax.set_ylabel("True", fontsize=7, color="black", weight="bold")
    ax.set_title("Confusion Matrix", fontsize=8, color="black", weight="bold", pad=5)
    ax.tick_params(axis="x", colors="black", labelsize=6)
    ax.tick_params(axis="y", colors="black", labelsize=6)
    ax.set_xticklabels(model.classes_, color="black", fontsize=6, weight="bold", rotation=0)
    ax.set_yticklabels(model.classes_, color="black", fontsize=6, weight="bold", rotation=0)
    st.pyplot(fig, use_container_width=False, bbox_inches="tight")

    # WordCloud
    st.subheader("☁️ Top 20 Kata Paling Sering Disebut")
    all_words = [word for tokens in df_model['Customer Review Clean'] for word in tokens]
    word_freq = nltk.FreqDist(all_words)
    wc = WordCloud(width=600, height=250, background_color="#f3f5e7", colormap="Dark2") \
        .generate_from_frequencies(dict(word_freq.most_common(20)))
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5), facecolor="#f3f5e7")
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2, use_container_width=False)

    # Prediksi manual
    st.markdown("---")
    st.subheader("🔮 Prediksi Review Baru")
    user_text = st.text_area("Masukkan Review 💬:")
    if st.button("Prediksi"):
        if user_text.strip():
            X_new = tfidf.transform([user_text])
            pred = model.predict(X_new)[0]
            color = "#006400" if pred == "Positive" else "#550000"
            st.markdown(
                f"""
                <div style="background-color:{color}; padding:10px; border-radius:8px; color:white; font-weight:bold;">
                    Sentimen Prediksi: {pred}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Tulis review terlebih dahulu sebelum prediksi.")

# ============================================================
# 3️⃣ CHATBOT TAB
# ============================================================
elif selected == "Chatbot":
    st.header("Chatbot Analisis Review (RAG-Based) 💬")
    if "Customer Review" not in df_model.columns:
        st.error("CSV harus mempunyai kolom *Customer Review*!")
        st.stop()

    # hapus jika masih ada duplikat review
    df_model = df_model.drop_duplicates(subset=["Customer Review"]).reset_index(drop=True)

    def row_to_text(row):
        return (
            f"Kategori: {row['Category']}. "
            f"Produk: {row['Product Name']}. "
            f"Lokasi: {row['Location']}. "
            f"Harga: Rp{row['Price']}. "
            f"Rating keseluruhan: {row['Overall Rating']}. "
            f"Jumlah terjual: {row['Number Sold']}. "
            f"Total review: {row['Total Review']}. "
            f"Rating pelanggan: {row['Customer Rating']}. "
            f"Review pelanggan: {row['Customer Review']}. "
            f"Sentimen: {row['Sentiment']}. "
            f"Emosi: {row['Emotion']}."
        )

    corpus = df_model.apply(row_to_text, axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    X_corpus = vectorizer.fit_transform(corpus)

    client = OpenAI(
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("samples"):
                st.markdown("Contoh Review Asli 📌:")
                for i, rev in enumerate(msg["samples"], 1):
                    st.markdown(f"{i}.** {rev}")

    if user_q := st.chat_input("Tulis Pertanyaan..."):
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        q_vec = vectorizer.transform([user_q])
        sim = cosine_similarity(q_vec, X_corpus).flatten()
        top_idx = sim.argsort()[-20:][::-1]
        retrieved_reviews = [corpus[i] for i in top_idx if sim[i] > 0.1]

        if not retrieved_reviews:
            answer = "❌ Tidak ditemukan jawaban yang relevan di review pelanggan."
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        else:
            context = "\n".join(retrieved_reviews)
            prompt = f"""
            Kamu adalah chatbot analisis review pelanggan.
            Jawablah pertanyaan user *hanya berdasarkan review di bawah ini*.
            Jangan menambah asumsi atau informasi dari luar.
            Sebutkan jumlah review relevan yang digunakan.
            Jika review saling bertentangan, sebutkan perbedaan pendapat pelanggan.

            Review terkait ({len(retrieved_reviews)} ditemukan):
            {context}

            Pertanyaan: {user_q}

            Jawaban (berikan ringkasan 1-2 paragraf):
            """
            with st.chat_message("assistant"):
                with st.spinner("Mohon tunggu sebentar, jawaban sedang diproses..."):
                    response = client.chat.completions.create(
                        model="deepseek/deepseek-chat-v3.1:free",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    answer = response.choices[0].message.content
                    sample_reviews = retrieved_reviews[:5]
                    st.write(answer)
                    st.markdown("📌 Contoh Review Asli:")
                    for i, rev in enumerate(sample_reviews, 1):
                        st.markdown(f"{i}.** {rev}")
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "samples": sample_reviews
                    })
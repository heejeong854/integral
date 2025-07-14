import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("🧠 딥러닝 손실 함수와 르베그 적분 시각화")

st.markdown("""
### 🎯 목표
딥러닝이 학습하는 핵심 목표는 '기대 손실(Expected Loss)'을 줄이는 것입니다.
이 기대 손실은 손실 함수와 데이터 분포의 **르베그 적분**으로 정의됩니다.
""")

# 입력 데이터 설정
x = np.linspace(0, 1, 500)
true_y = np.sin(2 * np.pi * x)
error_scale = st.slider("예측 오차 강도 조절 (예측이 틀린 정도)", 0.0, 2.0, 1.0)
pred_y = error_scale * true_y
loss = (true_y - pred_y)**2

# 분포 선택
dist_type = st.selectbox("입력 데이터의 분포 선택", ["균등분포 (Uniform)", "종모양 정규분포 (Normal)"])
if dist_type == "균등분포 (Uniform)":
    p_x = np.ones_like(x)
else:
    mu, sigma = 0.5, 0.15
    p_x = np.exp(-0.5 * ((x - mu) / sigma)**2)

p_x /= np.trapz(p_x, x)  # 정규화

# 기대 손실 = 손실 × 확률 밀도 적분
weighted_loss = loss * p_x
expected_loss = np.trapz(weighted_loss, x)

# 시각화
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, loss, color='red', label='손실 함수 L(x)')
ax.plot(x, p_x, color='blue', linestyle='--', label='입력 분포 p(x)')
ax.fill_between(x, weighted_loss, color='green', alpha=0.4, label='L(x) × p(x) (기여 면적)')

ax.set_xlabel("입력값 x")
ax.set_ylabel("함수값")
ax.set_title("손실 함수 × 데이터 분포 = 기대 손실 (르베그 적분)")
ax.legend()
st.pyplot(fig)

# 출력 결과
st.markdown(f"### ✅ 기대 손실 (Expected Loss): `{expected_loss:.4f}`")
st.markdown("→ 이 값이 바로 손실 함수와 확률 분포의 **르베그 적분** 결과입니다.")

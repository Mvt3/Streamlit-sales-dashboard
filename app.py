import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



# confg de la pág
st.set_page_config(
    page_title="Dashboard de análisis de ventas",
    page_icon="📊",
    layout="wide"
)

#Título
st.title("📊 Dashboard de análisis de ventas")
st.markdown("""
Este dashboard interactivo permite analizar datos de ventas, visualizar tendencias, 
realizar segmentación de clientes y predicciones de ventas futuras (Random forest).
""")

#Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        # Cargando csv
        df = pd.read_csv('Data/train.csv')
        return df
    except:
        st.info("Archivo de datos no encontrado.")
        return None
       
df = cargar_datos()


if df is not None:
    
    #convertir fechas a formato datetime con formato dd/mm/yyyy
    try:
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    except Exception as e:
        st.error(f"Error al convertir fechas: {e}")
        st.info("Intentando con formatos alternativos...")
        try:
            # Intenta con diferentes formatos comunes
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
            
            # Verificar si hay fechas no convertidas
            if df['Order Date'].isna().any():
                st.warning("Algunas fechas no pudieron ser convertidas correctamente. Revisar el formato de los datos.")
        except Exception as e:
            st.error(f"No se pudieron convertir las fechas: {e}")
            st.stop()
    
    # Separar fecha en componentes
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    
    #Menú lateral para navegación
    st.sidebar.title("Navegación aplicación")
    pagina = st.sidebar.radio(
        "Selecciona una sección:",
        ["Resumen general", "Análisis temporal", "Análisis geográfico",
         "Segmentación de clientes", "Análisis de productos", "Predicción de ventas"]
    )
    
    # Filtros generales **df_filtered será usado en todas las secciones**
    st.sidebar.title("Filtros")
    
    # Filtro de fechas 
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask_date = (df['Order Date'].dt.date >= start_date) & (df['Order Date'].dt.date <= end_date)
        df_filtered = df[mask_date]
    else:
        df_filtered = df
    
    #filtro de categoría
    categorias = ['Todas'] + list(df['Category'].unique())
    categoria_seleccionada = st.sidebar.selectbox("Categoría de producto", categorias)
    
    if categoria_seleccionada != 'Todas':
        df_filtered = df_filtered[df_filtered['Category'] == categoria_seleccionada]
    
    # Filtro de región
    regiones = ['Todas'] + list(df['Region'].unique())
    region_seleccionada = st.sidebar.selectbox("Región", regiones)
    
    if region_seleccionada != 'Todas':
        df_filtered = df_filtered[df_filtered['Region'] == region_seleccionada]
    
    # Verificar si hay datos después de aplicar filtros
    if len(df_filtered) == 0:
        st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros")
    else:
        # --------------- SECCIÓN: RESUMEN GENERAL ----------------
        if pagina == "Resumen general":
            st.header("📈 Resumen general de ventas")
            col1, col2, col3, col4 = st.columns(4)
        
            #Total de ventas
            total_sales = df_filtered['Sales'].sum()
            col1.metric("Ventas totales", f"${total_sales:,.2f}")
            
            #promedio de ventas por orden
            avg_sales_per_order = df_filtered.groupby('Order ID')['Sales'].sum().mean()
            col2.metric("Promedio por orden", f"${avg_sales_per_order:,.2f}")
            
            #número de pedidos
            num_orders = df_filtered['Order ID'].nunique()
            col3.metric("Número de pedidos", f"{num_orders:,}")
            
            #número de clientes
            num_customers = df_filtered['Customer ID'].nunique()
            col4.metric("Clientes únicos", f"{num_customers:,}")

            st.markdown("---")
            
            #gráficos de resumen
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Ventas por categoría")
                ventas_por_categoria = df_filtered.groupby('Category')['Sales'].sum().reset_index()
                fig = px.bar(
                    ventas_por_categoria, 
                    x='Category', 
                    y='Sales',
                    color='Category',
                    title="Ventas totales por categoría",
                    labels={'Sales': 'Ventas ($)', 'Category': 'Categoría'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Ventas por región")
                ventas_por_region = df_filtered.groupby('Region')['Sales'].sum().reset_index()
                fig = px.pie(
                    ventas_por_region, 
                    values='Sales', 
                    names='Region', 
                    title="Distribución de ventas por región",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tendencia temporal
            st.subheader("Tendencia de ventas")
            ventas_por_tiempo = df_filtered.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
            fig = px.line(
                ventas_por_tiempo, 
                x='Order Date', 
                y='Sales',
                title="Ventas mensuales ",
                labels={'Sales': 'Ventas ($)', 'Order Date': 'Fecha'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de resumen
            st.subheader("Resumen de ventas por subcategoría")
            resumen_subcategoria = df_filtered.groupby(['Category', 'Sub-Category'])['Sales'].agg(
                ['sum', 'mean', 'count']
            ).reset_index()
            resumen_subcategoria.columns = ['Categoría', 'Subcategoría', 'Ventas Totales', 'Venta Promedio', 'Número de Ventas']
            resumen_subcategoria['Ventas Totales'] = resumen_subcategoria['Ventas Totales'].map('${:,.2f}'.format)
            resumen_subcategoria['Venta Promedio'] = resumen_subcategoria['Venta Promedio'].map('${:,.2f}'.format)
            st.dataframe(resumen_subcategoria, use_container_width=True)
    
        # ----------------- SECCIÓN: ANÁLISIS TEMPORAL -----------------
        elif pagina == "Análisis temporal":
            st.header("⏱️ Análisis temporal de ventas")
            
            # selección de nivel de tiempo
            nivel_tiempo = st.radio(
                "Selecciona el nivel de tiempo para el análisis:",
                ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"],
                horizontal=True
            )
            
            # función para crear datos agregados por tiempo
            def agregar_por_tiempo(df, nivel):
                if nivel == "Diario":
                    return df.groupby(df['Order Date'].dt.date)['Sales'].sum().reset_index()
                elif nivel == "Semanal":
                    return df.groupby(pd.Grouper(key='Order Date', freq='W-MON'))['Sales'].sum().reset_index()
                elif nivel == "Mensual":
                    return df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
                elif nivel == "Trimestral":
                    return df.groupby(pd.Grouper(key='Order Date', freq='Q'))['Sales'].sum().reset_index()
                else:  # Anual
                    return df.groupby(pd.Grouper(key='Order Date', freq='Y'))['Sales'].sum().reset_index()
            
            ventas_tiempo = agregar_por_tiempo(df_filtered, nivel_tiempo)
            
            # gráfico de líneas de ventas a lo largo del tiempo
            st.subheader(f"Tendencia de ventas ({nivel_tiempo})")
            fig = px.line(
                ventas_tiempo, 
                x='Order Date', 
                y='Sales',
                title=f"Ventas {nivel_tiempo}s",
                labels={'Sales': 'Ventas ($)', 'Order Date': 'Fecha'}
            )
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de estacionalidad
            if nivel_tiempo in ["Mensual", "Trimestral", "Anual"]:
                st.subheader("Análisis de estacionalidad")
                
                ventas_mes = df_filtered.copy()
                ventas_mes['Mes'] = ventas_mes['Order Date'].dt.month_name()
                ventas_por_mes = ventas_mes.groupby('Mes')['Sales'].sum().reset_index()
                
                #Orden meses cronológicamente
                meses_orden = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                ventas_por_mes['Mes'] = pd.Categorical(ventas_por_mes['Mes'], categories=meses_orden, ordered=True)
                ventas_por_mes = ventas_por_mes.sort_values('Mes')
                
                fig = px.bar(
                    ventas_por_mes, 
                    x='Mes', 
                    y='Sales',
                    color='Sales',
                    color_continuous_scale='Viridis',
                    title="Estacionalidad de ventas por mes",
                    labels={'Sales': 'Ventas ($)', 'Mes': 'Mes'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                #Análisis por día de la semana
                st.subheader("Ventas por día de la semana")
                ventas_mes['DiaSemana'] = ventas_mes['Order Date'].dt.day_name()
                ventas_por_dia = ventas_mes.groupby('DiaSemana')['Sales'].sum().reset_index()

                # Orden cronológico
                dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                ventas_por_dia['DiaSemana'] = pd.Categorical(ventas_por_dia['DiaSemana'], categories=dias_orden, ordered=True)
                ventas_por_dia = ventas_por_dia.sort_values('DiaSemana')
                
                fig = px.bar(
                    ventas_por_dia, 
                    x='DiaSemana', 
                    y='Sales',
                    color='Sales',
                    color_continuous_scale='Viridis',
                    title="Ventas por día de la semana",
                    labels={'Sales': 'Ventas ($)', 'DiaSemana': 'Día de la semana'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de crecimiento
            st.subheader("Análisis de crecimiento")

            if nivel_tiempo in ["Diario","Semanal" ,"Mensual", "Trimestral", "Anual"]:
                #Calculo de crecimiento
                ventas_tiempo['Ventas Previas'] = ventas_tiempo['Sales'].shift(1)
                ventas_tiempo['Crecimiento'] = (ventas_tiempo['Sales'] - ventas_tiempo['Ventas Previas']) / ventas_tiempo['Ventas Previas'] * 100
                ventas_tiempo.fillna({'Crecimiento': 0}, inplace=True)
                
                #Mostrar tabla 
                tabla_crecimiento = ventas_tiempo[['Order Date', 'Sales', 'Crecimiento']].copy()
                tabla_crecimiento.columns = ['Fecha', 'Ventas', 'Crecimiento (%)']
                tabla_crecimiento['Ventas'] = tabla_crecimiento['Ventas'].map('${:,.2f}'.format)
                tabla_crecimiento['Crecimiento (%)'] = tabla_crecimiento['Crecimiento (%)'].map('{:,.2f}%'.format)
                
                st.dataframe(tabla_crecimiento, use_container_width=True)
    
        # ---------------- SECCIÓN: ANÁLISIS GEOGRÁFICO ----------------
        elif pagina == "Análisis geográfico":
            st.header("🗺️ Análisis geográfico de ventas (USA)")
            
            # Diccionario para convertir nombres completos de estados a abreviaturas
            us_state_to_abbrev = {
                "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
                "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
                "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
                "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
                "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
                "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
                "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
                "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
                "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
                "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
                "District of Columbia": "DC"
            }
          
            st.subheader("Distribución Geográfica de Ventas")
            
            # Preprocesar los datos para el mapa
            ventas_por_estado = df_filtered.groupby(['State'])['Sales'].sum().reset_index()
            

            sample_state = ventas_por_estado['State'].iloc[0] if not ventas_por_estado.empty else ""
            
            # Si los estados están en formato de nombre completo, convertirlos a códigos
            if len(sample_state) > 2:
                ventas_por_estado['state_code'] = ventas_por_estado['State'].map(us_state_to_abbrev)
                # Para visualizar el mapa necesitamos los códigos de estado
                fig_map_data = ventas_por_estado[['state_code', 'Sales']].rename(columns={'state_code': 'State'})
           
            
            # Crear visualización con los datos procesados
            try:
                fig = px.choropleth(
                    fig_map_data,
                    locations='State',
                    locationmode='USA-states',
                    color='Sales',
                    scope='usa',
                    color_continuous_scale='Viridis',
                    labels={'Sales': 'Ventas ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar el mapa de estados: {e}")
                st.info("Mostrando visualización alternativa de estados...")
                
                # Alternativa: gráfico de barras
                ventas_por_estado_sorted = ventas_por_estado.sort_values(by='Sales', ascending=False)
                fig = px.bar(
                    ventas_por_estado_sorted.head(15), 
                    x='State', 
                    y='Sales',
                    title="Top 15 estados por vntas",
                    labels={'Sales': 'Ventas ($)', 'State': 'Estado'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ventas por región
            st.subheader("Ventas por región")
            ventas_por_region = df_filtered.groupby('Region')['Sales'].sum().reset_index()
            fig = px.bar(
                ventas_por_region, 
                x='Region', 
                y='Sales',
                color='Region',
                title="Ventas totales por región",
                labels={'Sales': 'Ventas ($)', 'Region': 'Región'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top ciudades
            st.subheader("Top ciudades por ventas")
            
            num_ciudades = st.slider("Seleccionar número de ciudades a mostrar:", 5, 10, 20)
            
            top_ciudades = df_filtered.groupby('City')['Sales'].sum().reset_index()
            top_ciudades = top_ciudades.sort_values(by='Sales', ascending=False).head(num_ciudades)
            
            fig = px.bar(
                top_ciudades, 
                x='City', 
                y='Sales',
                color='Sales',
                color_continuous_scale='Viridis',
                title=f"Top {num_ciudades} Ciudades por ventas",
                labels={'Sales': 'Ventas ($)', 'City': 'Ciudad'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            #Tablas de resumen por región y estado
            st.subheader("Resumen de ventas por región y estado")
            resumen_geo = df_filtered.groupby(['Region', 'State'])['Sales'].agg(
                ['sum', 'mean', 'count']
            ).reset_index()
            resumen_geo.columns = ['Región', 'Estado', 'Ventas Totales', 'Venta Promedio', 'Número de Ventas']
            resumen_geo = resumen_geo.sort_values(by=['Región', 'Ventas Totales'], ascending=[True, False])
            resumen_geo['Ventas Totales'] = resumen_geo['Ventas Totales'].map('${:,.2f}'.format)
            resumen_geo['Venta Promedio'] = resumen_geo['Venta Promedio'].map('${:,.2f}'.format)
        
            st.dataframe(resumen_geo, use_container_width=True)
           
    
        # ------------------ SECCIÓN: SEGMENTACIÓN DE CLIENTES ----------------
        elif pagina == "Segmentación de clientes":
            st.header("👥 Segmentación de clientes")
            
            
            st.subheader("Segmentación por valor de cliente")
            
            # Calcular métricas por cliente
            clientes_metricas = df_filtered.groupby('Customer ID').agg({
                'Sales': ['sum', 'mean', 'count'],
                'Order ID': 'nunique',
                'Customer Name': 'first'
            })
            
            clientes_metricas.columns = ['Ventas Totales', 'Venta Promedio', 'Número de Items', 'Número de Pedidos', 'Nombre Cliente']
            clientes_metricas['Frecuencia'] = clientes_metricas['Número de Pedidos'] / clientes_metricas.index.nunique()
            clientes_metricas = clientes_metricas.reset_index()
            
            # Segmentar clientes por valor 
            def segmentar_valor(row):
                if row['Ventas Totales'] > clientes_metricas['Ventas Totales'].quantile(0.75):
                    return "Alto Valor"
                elif row['Ventas Totales'] > clientes_metricas['Ventas Totales'].quantile(0.5):
                    return "Valor Medio-Alto"
                elif row['Ventas Totales'] > clientes_metricas['Ventas Totales'].quantile(0.25):
                    return "Valor Medio-Bajo"
                else:
                    return "Bajo Valor"
            
            clientes_metricas['Segmento Valor'] = clientes_metricas.apply(segmentar_valor, axis=1)
            
            # Visualizar distribución de segmentos
            fig = px.pie(
                clientes_metricas, 
                names='Segmento Valor', 
                title="Distribución de clientes por segmento de valor",
                color='Segmento Valor',
                color_discrete_map={
                    'Alto Valor': '#636EFA',
                    'Valor Medio-Alto': '#EF553B',
                    'Valor Medio-Bajo': '#00CC96',
                    'Bajo Valor': '#AB63FA'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de dispersión frecuencia vs valor
            st.subheader("Relación entre frecuencia y valor de cliente")
            fig = px.scatter(
                clientes_metricas,
                x='Número de Pedidos',
                y='Ventas Totales',
                color='Segmento Valor',
                hover_name='Nombre Cliente',
                size='Venta Promedio',
                opacity=0.7,
                title="Matriz de valor vs. frecuencia de pedidos",
                labels={
                    'Número de Pedidos': 'Frecuencia (# Pedidos)',
                    'Ventas Totales': 'Valor Total ($)',
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis por segmento de cliente
            st.subheader("Ventas por segmento de cliente")
            ventas_por_segmento = df_filtered.groupby('Segment')['Sales'].sum().reset_index()
            fig = px.bar(
                ventas_por_segmento, 
                x='Segment', 
                y='Sales',
                color='Segment',
                title="Ventas totales por segmento",
                labels={'Sales': 'Ventas ($)', 'Segment': 'Segmento'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top clientes
            st.subheader("Top clientes por ventas")
            
            num_clientes = st.slider("Número de clientes a mostrar:", 5, 20, 10)
            
            top_clientes = clientes_metricas.sort_values(by='Ventas Totales', ascending=False).head(num_clientes)
            top_clientes['Ventas Totales Formato'] = top_clientes['Ventas Totales'].map('${:,.2f}'.format)
            top_clientes['Venta Promedio Formato'] = top_clientes['Venta Promedio'].map('${:,.2f}'.format)
            
            fig = px.bar(
                top_clientes, 
                x='Nombre Cliente', 
                y='Ventas Totales',
                text='Ventas Totales Formato',
                color='Segmento Valor',
                title=f"Top {num_clientes} Clientes por Ventas",
                labels={'Ventas Totales': 'Ventas ($)', 'Nombre Cliente': 'Cliente'}
            )
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
    
        # ----------------- SECCIÓN: ANÁLISIS DE PRODUCTOS -----------------
        elif pagina == "Análisis de productos":
            st.header("🛒 Análisis de productos")
            
            # Top productos por ventas
            st.subheader("Top productos por ventas")
            
            num_productos = st.slider("Número de productos a mostrar:", 5, 20, 10)
            
            top_productos = df_filtered.groupby('Product Name')['Sales'].sum().reset_index()
            top_productos = top_productos.sort_values(by='Sales', ascending=False).head(num_productos)
            
            #bar h
            fig = px.bar(
                top_productos, 
                x='Sales', 
                y='Product Name',
                orientation='h',
                color='Sales',
                color_continuous_scale='Viridis',
                title=f"Top {num_productos} Productos por Ventas",
                labels={'Sales': 'Ventas ($)', 'Product Name': 'Producto'}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            #Análisis por subcategoría
            st.subheader("Ventas por subcategoría")
            
            ventas_subcategoria = df_filtered.groupby(['Category', 'Sub-Category'])['Sales'].sum().reset_index()
            
            fig = px.treemap(
                ventas_subcategoria,
                path=['Category', 'Sub-Category'],
                values='Sales',
                color='Sales',
                color_continuous_scale='Viridis',
                title="Distribución de ventas por categoría y subcategoría"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de tendencias por categoría
            st.subheader("Tendencia de ventas por categoría")
            ventas_categoria_tiempo = df_filtered.groupby([pd.Grouper(key='Order Date', freq='M'), 'Category'])['Sales'].sum().reset_index()
            fig = px.line(
                ventas_categoria_tiempo, 
                x='Order Date', 
                y='Sales',
                color='Category',
                title="Tendencia de ventas mensuales por categoría",
                labels={'Sales': 'Ventas ($)', 'Order Date': 'Fecha', 'Category': 'Categoría'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            #Tabla de resumen de productos
            st.subheader("Resumen de ventas por producto")
            
            resumen_productos = df_filtered.groupby('Product Name')['Sales'].agg(
                ['sum', 'mean', 'count']
            ).reset_index()

            resumen_productos.columns = ['Producto', 'Ventas Totales', 'Venta Promedio', 'Número de Ventas']
            resumen_productos = resumen_productos.sort_values(by='Ventas Totales', ascending=False)
            resumen_productos['Ventas Totales'] = resumen_productos['Ventas Totales'].map('${:,.2f}'.format)
            resumen_productos['Venta Promedio'] = resumen_productos['Venta Promedio'].map('${:,.2f}'.format)
        
            st.dataframe(resumen_productos, use_container_width=True)
    
        # ----------------- SECCIÓN: PREDICCIÓN DE VENTAS -----------------
        elif pagina == "Predicción de ventas":
            st.header("Predicción de ventas")
            
            st.info("Esta sección utiliza un modelo de random forest para predecir ventas futuras basadas en patrones históricos.")
            df_model = df_filtered.copy()
            
            # Crear características temporales
            df_model['Year'] = df_model['Order Date'].dt.year
            df_model['Month'] = df_model['Order Date'].dt.month
            df_model['Day'] = df_model['Order Date'].dt.day
            df_model['DayOfWeek'] = df_model['Order Date'].dt.dayofweek
            df_model['Quarter'] = df_model['Order Date'].dt.quarter
            
           
            st.subheader("Selecciona periodo para predicción")
            
            periodo = st.radio(
                "Nivel de agregación para la predicción:",
                ["Mensual", "Trimestral"],
                horizontal=True
            )
            
            # Función para agregar por periodo
            if periodo == "Mensual":
                ventas_periodo = df_model.groupby([df_model['Year'], df_model['Month']])['Sales'].sum().reset_index()
                ventas_periodo['Period'] = ventas_periodo['Year'].astype(str) + '-' + ventas_periodo['Month'].astype(str).str.zfill(2)
                ordenes = df_model.groupby(['Year', 'Month'])['Order ID'].nunique().reset_index(name='OrderCount')
                productos = df_model.groupby(['Year', 'Month'])['Product ID'].nunique().reset_index(name='ProductCount')
                ventas_periodo = ventas_periodo.merge(ordenes, on=['Year', 'Month'])
                ventas_periodo = ventas_periodo.merge(productos, on=['Year', 'Month'])
                ventas_periodo['PeriodNum'] = range(1, len(ventas_periodo) + 1)

            else:
                ventas_periodo = df_model.groupby([df_model['Year'], df_model['Quarter']])['Sales'].sum().reset_index()
                ventas_periodo['Period'] = ventas_periodo['Year'].astype(str) + '-Q' + ventas_periodo['Quarter'].astype(str)
                ordenes = df_model.groupby(['Year', 'Quarter'])['Order ID'].nunique().reset_index(name='OrderCount')
                productos = df_model.groupby(['Year', 'Quarter'])['Product ID'].nunique().reset_index(name='ProductCount')
                ventas_periodo = ventas_periodo.merge(ordenes, on=['Year', 'Quarter'])
                ventas_periodo = ventas_periodo.merge(productos, on=['Year', 'Quarter'])
                ventas_periodo['PeriodNum'] = range(1, len(ventas_periodo) + 1)

            # Construir features y etiquetas
            col_tiempo = 'Month' if periodo == "Mensual" else 'Quarter'
            X = ventas_periodo[['PeriodNum', 'Year', col_tiempo, 'OrderCount', 'ProductCount']]
            y = ventas_periodo['Sales']

            # Modelo simple de rf
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                #Entrenar modelo, valores fueron escogidos con el uso de grid search. *todo eso lo hice fuera de este archivo, como tambien un cross-v para verificar overfitting*
                model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2,random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Mostrar métricas
                col1, col2, col3 = st.columns(3)
                col1.metric("Error Absoluto Medio", f"${mae:,.2f}")
                col2.metric("RMSE", f"${rmse:,.2f}")
                col3.metric("R² Score", f"{r2:.4f}")
                
                #Realizar predicciones para periodos futuros
                st.subheader("Predicción de ventas futuras")
                
                num_periodos = st.slider("Número de periodos a predecir:", 1, 6, 12)
                
                # Crear periodos futuros
                last_period = ventas_periodo.iloc[-1]
                future_periods = []
                
                for i in range(1, num_periodos + 1):
                    new_period = {}
                    new_period['PeriodNum'] = last_period['PeriodNum'] + i
                    
                    if periodo == "Mensual":
                        new_month = (last_period['Month'] + i - 1) % 12 + 1
                        new_year = last_period['Year'] + (last_period['Month'] + i - 1) // 12
                        new_period['Year'] = new_year
                        new_period['Month'] = new_month
                        new_period['Period'] = f"{new_year}-{new_month:02d}"
                        order_count = df_model[(df_model['Year'] == new_year) & (df_model['Month'] == new_month)]['Order ID'].nunique()
                        product_count = df_model[(df_model['Year'] == new_year) & (df_model['Month'] == new_month)]['Product ID'].nunique()
                        new_period['OrderCount'] = order_count
                        new_period['ProductCount'] = product_count

                    else:  # Trimestral
                        new_quarter = (last_period['Quarter'] + i - 1) % 4 + 1
                        new_year = last_period['Year'] + (last_period['Quarter'] + i - 1) // 4
                        new_period['Year'] = new_year
                        new_period['Quarter'] = new_quarter
                        new_period['Period'] = f"{new_year}-Q{new_quarter}"
                        order_count = df_model[(df_model['Year'] == new_year) & (df_model['Quarter'] == new_quarter)]['Order ID'].nunique()
                        product_count = df_model[(df_model['Year'] == new_year) & (df_model['Quarter'] == new_quarter)]['Product ID'].nunique()
                        new_period['OrderCount'] = order_count
                        new_period['ProductCount'] = product_count

                    future_periods.append(new_period)
                
                future_df = pd.DataFrame(future_periods)
                
                # Preparar para predicción
                if periodo == "Mensual":
                    X_future = future_df[['PeriodNum', 'Year', 'Month','OrderCount', 'ProductCount']]
                else:  # Trimestral
                    X_future = future_df[['PeriodNum', 'Year', 'Quarter','OrderCount', 'ProductCount']]

                # Predecir ventas futuras
                future_sales = model.predict(X_future)
                future_df['Predicted_Sales'] = future_sales
                
                # Combinar datos históricos y predicciones para visualización
                historical_data = ventas_periodo[['Period', 'Sales']].copy()
                historical_data['Tipo'] = 'Histórico'
                
                forecast_data = future_df[['Period', 'Predicted_Sales']].copy()
                forecast_data.rename(columns={'Predicted_Sales': 'Sales'}, inplace=True)
                forecast_data['Tipo'] = 'Predicción'
                
                combined_data = pd.concat([historical_data, forecast_data], ignore_index=True)
                
                # Visualizar predicciones
                fig = px.line(
                    combined_data,
                    x='Period',
                    y='Sales',
                    color='Tipo',
                    title="Ventas históricas y predicción",
                    labels={'Sales': 'Ventas ($)', 'Period': 'Periodo'},
                    color_discrete_map={'Histórico': '#636EFA', 'Predicción': '#EF553B'}
                )
                fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray': combined_data['Period'].tolist()})
                st.plotly_chart(fig, use_container_width=True)
                
                #tabla de predicciones
                st.subheader("Detalle de predicciones")
                forecast_display = future_df[['Period', 'Predicted_Sales']].copy()
                forecast_display.columns = ['Periodo', 'Ventas Previstas']
                forecast_display['Ventas Previstas'] = forecast_display['Ventas Previstas'].map('${:,.2f}'.format)
                st.dataframe(forecast_display, use_container_width=True)
                
                #variables de importancia
                st.subheader("Factores que influyen en las predicciones")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Importancia de características en el modelo",
                    labels={'Importance': 'Importancia', 'Feature': 'Característica'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ocurrió un error al crear el modelo de predicción: {e}")
                st.info("Para obtener predicciones precisas, asegúrate de tener suficientes datos históricos disponibles.")








// KRATOS ___ ___  _  ___   __   ___ ___ ___ ___
//       / __/ _ \| \| \ \ / /__|   \_ _| __| __|
//      | (_| (_) | .` |\ V /___| |) | || _|| _|
//       \___\___/|_|\_| \_/    |___/___|_| |_|  APPLICATION
//
//  License: BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:  Julio Marti & Miguel Angel Celigueta
//

#if !defined(KRATOS_EULERIAN_CONVECTION_DIFFUSION_LUMPED_ELEMENT_INCLUDED )
#define  KRATOS_EULERIAN_CONVECTION_DIFFUSION_LUMPED_ELEMENT_INCLUDED


// System includes


// External includes


// Project includes
#include "../../ConvectionDiffusionApplication/custom_elements/eulerian_conv_diff.h"

namespace Kratos
{

template< unsigned int TDim, unsigned int TNumNodes>
class EulerianConvectionDiffusionLumpedElement
    : public EulerianConvectionDiffusionElement<TDim,TNumNodes>
{

public:
    typedef Node < 3 > NodeType;
    typedef Geometry <NodeType>   GeometryType;
    typedef Properties PropertiesType;
    typedef Geometry<NodeType>::PointsArrayType NodesArrayType;
    typedef Vector VectorType;
    typedef Matrix MatrixType;


    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(EulerianConvectionDiffusionLumpedElement);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// Default constructor.

    EulerianConvectionDiffusionLumpedElement() : EulerianConvectionDiffusionElement<TDim, TNumNodes>()
    {}

    EulerianConvectionDiffusionLumpedElement(IndexType NewId, GeometryType::Pointer pGeometry)
    : EulerianConvectionDiffusionElement<TDim, TNumNodes>(NewId, pGeometry)
    {}

    EulerianConvectionDiffusionLumpedElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties)
    : EulerianConvectionDiffusionElement<TDim, TNumNodes>(NewId, pGeometry, pProperties)
    {}

    /// Destructor.
    virtual ~EulerianConvectionDiffusionLumpedElement() {};

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    Element::Pointer Create(
        IndexType NewId,
        NodesArrayType const& ThisNodes,
        PropertiesType::Pointer pProperties
        ) const override
    {
        return Kratos::make_intrusive<EulerianConvectionDiffusionLumpedElement>(NewId, this->GetGeometry().Create(ThisNodes), pProperties);
    }

    Element::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties
        ) const override
    {
        return Kratos::make_intrusive<EulerianConvectionDiffusionLumpedElement>(NewId, pGeom, pProperties);
    }

    void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override {
        KRATOS_TRY

        // Resize of the Left and Right Hand side
        if (rLeftHandSideMatrix.size1() != TNumNodes)
            rLeftHandSideMatrix.resize(TNumNodes, TNumNodes, false); //false says not to preserve existing storage!!
        noalias(rLeftHandSideMatrix) = ZeroMatrix(TNumNodes, TNumNodes);

        if (rRightHandSideVector.size() != TNumNodes)
            rRightHandSideVector.resize(TNumNodes, false); //false says not to preserve existing storage!!
        noalias(rRightHandSideVector) = ZeroVector(TNumNodes);

        //Element variables
        typename EulerianConvectionDiffusionElement<TDim, TNumNodes>::ElementVariables Variables;
        this->InitializeEulerianElement(Variables,rCurrentProcessInfo);

        // Compute the geometry
        BoundedMatrix<double,TNumNodes, TDim> DN_DX;
        array_1d<double,TNumNodes > N;
        double Volume;
        this-> CalculateGeometry(DN_DX,Volume);

        // Getting the values of shape functions on Integration Points
        BoundedMatrix<double,TNumNodes, TNumNodes> Ncontainer;
        const GeometryType& Geom = this->GetGeometry();
        Ncontainer = Geom.ShapeFunctionsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 );

        // Getting the values of Current Process Info and computing the value of h
        this-> GetNodalValues(Variables,rCurrentProcessInfo);
        double h = this->ComputeH(DN_DX);

        //Computing the divergence
        for (unsigned int i = 0; i < TNumNodes; i++)
        {
            for(unsigned int k=0; k<TDim; k++)
            {
                Variables.div_v += DN_DX(i,k)*(Variables.v[i][k]*Variables.theta + Variables.vold[i][k]*(1.0-Variables.theta));
            }
        }

        //Some auxilary definitions
        BoundedMatrix<double,TNumNodes, TNumNodes> aux1 = ZeroMatrix(TNumNodes, TNumNodes); //terms multiplying dphi/dt
        BoundedMatrix<double,TNumNodes, TNumNodes> aux2 = ZeroMatrix(TNumNodes, TNumNodes); //terms multiplying phi
        bounded_matrix<double,TNumNodes, TDim> tmp;

        unsigned int NumGPoints = Geom.IntegrationPointsNumber( GeometryData::IntegrationMethod::GI_GAUSS_2 );
        array_1d<double,TNumNodes> Integrity = ZeroVector(TNumNodes); // Here TNumNodes = NumGPoints
        for ( unsigned int igauss = 0; igauss < NumGPoints; igauss++ )
        {
            for ( unsigned int i = 0; i < TNumNodes; i++ )
            {
                // TODO. This needs to be reviewed...
                // Integrity[igauss] += Ncontainer(igauss,i)*(1.0 - Geom[i].FastGetSolutionStepValue(DECOMPOSITION));
                Integrity[igauss] += Ncontainer(igauss,i)*(1.0 - 0.0);
            }
        }
        // TODO. Testing
        // if(rCurrentProcessInfo[STEP] == 1){
        //     noalias(Integrity) = ZeroVector(TNumNodes);
        //     for ( unsigned int igauss = 0; igauss < NumGPoints; igauss++ )
        //     {
        //         for ( unsigned int i = 0; i < TNumNodes; i++ )
        //         {
        //             Integrity[igauss] += Ncontainer(igauss,i)*(1.0 - 0.0);
        //         }
        //     }
        // }
        // if(rCurrentProcessInfo[STEP] == 2){
        //     noalias(Integrity) = ZeroVector(TNumNodes);
        //     for ( unsigned int igauss = 0; igauss < NumGPoints; igauss++ )
        //     {
        //         for ( unsigned int i = 0; i < TNumNodes; i++ )
        //         {
        //             Integrity[igauss] += Ncontainer(igauss,i)*(1.0 - 0.5);
        //         }
        //     }
        // }

        // Gauss points and Number of nodes coincides in this case.
        double integrity_quadrature = 0.0;

        for(unsigned int igauss=0; igauss<TNumNodes; igauss++)
        {
            noalias(N) = row(Ncontainer,igauss);

            integrity_quadrature += Integrity[igauss];

            //obtain the velocity in the middle of the time step
            array_1d<double, TDim > vel_gauss=ZeroVector(TDim);
            for (unsigned int i = 0; i < TNumNodes; i++)
            {
                 for(unsigned int k=0; k<TDim; k++)
                    vel_gauss[k] += N[i]*(Variables.v[i][k]*Variables.theta + Variables.vold[i][k]*(1.0-Variables.theta));
            }
            const double norm_vel = norm_2(vel_gauss);
            array_1d<double, TNumNodes > a_dot_grad = prod(DN_DX, vel_gauss);

            const double tau = this->CalculateTau(Variables,norm_vel,h);

            //terms multiplying dphi/dt (aux1)
            noalias(aux1) += (1.0*Integrity[igauss]+tau*Variables.beta*Variables.div_v)* 0.25 * IdentityMatrix(4, 4); //outer_prod(N, N); //0.25 * IdentityMatrix(4, 4);
            noalias(aux1) +=  tau*outer_prod(a_dot_grad, N);
            // aux1 *= Integrity[igauss];

            //terms which multiply the gradient of phi
            noalias(aux2) += (1.0*Integrity[igauss]+tau*Variables.beta*Variables.div_v)*outer_prod(N, a_dot_grad);
            noalias(aux2) += tau*outer_prod(a_dot_grad, a_dot_grad);
            // aux2 *= Integrity[igauss];
        }

        //adding the second and third term in the formulation
        noalias(rLeftHandSideMatrix)  += (Variables.dt_inv*Variables.density*Variables.specific_heat + Variables.theta*Variables.beta*Variables.div_v)*aux1;
        noalias(rRightHandSideVector) += (Variables.dt_inv*Variables.density*Variables.specific_heat - (1.0-Variables.theta)*Variables.beta*Variables.div_v)*prod(aux1,Variables.phi_old);

        //adding the diffusion
        const double effective_conductivity = Variables.conductivity;
        noalias(rLeftHandSideMatrix)  += (effective_conductivity * Variables.theta * prod(DN_DX, trans(DN_DX)))*static_cast<double>(TNumNodes);
        noalias(rRightHandSideVector) -= prod((effective_conductivity * (1.0-Variables.theta) * prod(DN_DX, trans(DN_DX))),Variables.phi_old)*static_cast<double>(TNumNodes) ;

        //terms in aux2
        noalias(rLeftHandSideMatrix) += Variables.density * Variables.specific_heat*Variables.theta*aux2;
        noalias(rRightHandSideVector) -= Variables.density * Variables.specific_heat*(1.0-Variables.theta)*prod(aux2,Variables.phi_old);

        // volume source terms (affecting the RHS only)
        noalias(rRightHandSideVector) += prod(aux1, Variables.volumetric_source);

        //take out the dirichlet part to finish computing the residual
        noalias(rRightHandSideVector) -= prod(rLeftHandSideMatrix, Variables.phi);

        rRightHandSideVector *= Volume/static_cast<double>(TNumNodes);
        rLeftHandSideMatrix *= Volume/static_cast<double>(TNumNodes);

        // TODO. Testing
        // if((rCurrentProcessInfo[STEP] == 1) && (this->Id()==1846)){
        //     KRATOS_WATCH(this->Id())
        //     KRATOS_WATCH(integrity_quadrature/4.0)
        //     KRATOS_WATCH(rRightHandSideVector)
        //     KRATOS_WATCH(rLeftHandSideMatrix)
        // }
        // if((rCurrentProcessInfo[STEP] == 2) && (this->Id()==1846)){
        //     KRATOS_WATCH(this->Id())
        //     KRATOS_WATCH(integrity_quadrature/4.0)
        //     KRATOS_WATCH(rRightHandSideVector)
        //     KRATOS_WATCH(rLeftHandSideMatrix)
        // }

        KRATOS_CATCH("Error in Eulerian ConvDiff Lumped Element")
    }

    void CalculateOnIntegrationPoints( const Variable<double>& rVariable,std::vector<double>& rValues,
                                                const ProcessInfo& rCurrentProcessInfo ) override
    {
        if(rVariable == DECOMPOSITION)
        {
            const GeometryType& Geom = this->GetGeometry();
            const unsigned int NumGPoints = Geom.IntegrationPointsNumber( GeometryData::IntegrationMethod::GI_GAUSS_2 );
            const Matrix& NContainer = Geom.ShapeFunctionsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 );

            if ( rValues.size() != NumGPoints )
                rValues.resize(NumGPoints);

            for ( unsigned int igauss = 0; igauss < NumGPoints; igauss++ )
            {
                double gp_decomposition = 0.0;
                for ( unsigned int i = 0; i < TNumNodes; i++ )
                {
                    gp_decomposition += NContainer(igauss,i)*Geom[i].FastGetSolutionStepValue(DECOMPOSITION);
                }
                rValues[igauss] = gp_decomposition;
            }
        }
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    std::string Info() const override
    {
        return "EulerianConvectionDiffusionLumpedElement #";
    }

    /// Print information about this object.

    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << Info() << this->Id();
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

protected:

    // Member Variables


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

private:

        // Serialization

    friend class Serializer;

    void save(Serializer& rSerializer) const override
    {
        KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element);

    }

    void load(Serializer& rSerializer) override
    {
        KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element);

    }


}; // Class EulerianConvectionDiffusionLumpedElement

} // namespace Kratos.

#endif // KRATOS_EULERIAN_CONVECTION_DIFFUSION_LUMPED_ELEMENT_INCLUDED  defined
